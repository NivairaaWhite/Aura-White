"""AuraNative: the shape VAE + the latent flow generator + their latent statistics, with save / load / generate.

    model = AuraNative.new("tiny")                         # or AuraNative.load("champion.pt")
    out = model.generate([ViewIn(picture, az=0), ViewIn(back_picture, az=180)], n=4, seed=1)[0]
    out.verts, out.faces, out.colours                      # a closed mesh with colours in the picture's own frame

Frame: the FIRST view is the front (+z, y up); other views give their azimuth relative to it (a back picture is az=180).
Checkpoints are plain tensors + JSON (loaded with `weights_only=True`: no pickle code can run).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import geometry as G
from .config import NativeConfig, preset
from .flow import LatentFlow, sample_latents
from .vae import ShapeVAE, count_params

FORMAT = "aura-native-1"


@dataclass
class ViewIn:
    image: np.ndarray            # (S,S,4) float32 in [0,1]: rgb composited over grey, 4th channel = silhouette (render.prep_picture)
    az: float = 0.0              # degrees, relative to the first view
    el: float = 0.0


@dataclass
class Generated:
    verts: np.ndarray
    faces: np.ndarray
    colours: np.ndarray
    seed: int
    z: Optional[np.ndarray] = None           # normalised latent grid (C,G,G,G): what the generator drew
    info: Dict = field(default_factory=dict)


def img_to_tensor(im: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(im, dtype=np.float32)).permute(2, 0, 1)
    if t.shape[-1] != size:
        t = F.interpolate(t[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0]
    return t * 2.0 - 1.0


class AuraNative:
    def __init__(self, cfg: NativeConfig, device="cpu"):
        self.cfg = cfg
        self.vae = ShapeVAE(cfg.vae)
        self.flow = LatentFlow(cfg.vae, cfg.flow)
        C = cfg.vae.latent_dim
        self.mean = torch.zeros(C)
        self.std = torch.ones(C)
        self.card: Dict = {"format": FORMAT, "preset": cfg.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "history": []}
        self.guidance_distilled = False
        self.device = torch.device("cpu")
        self.to(device)

    # -- construction / IO
    @classmethod
    def new(cls, name: str = "tiny", device="cpu", seed: int = 0) -> "AuraNative":
        torch.manual_seed(seed)
        return cls(preset(name), device)

    def to(self, device) -> "AuraNative":
        self.device = torch.device(device)
        self.vae.to(self.device)
        self.flow.to(self.device)
        self.mean, self.std = self.mean.to(self.device), self.std.to(self.device)
        return self

    def eval(self) -> "AuraNative":
        self.vae.eval()
        self.flow.eval()
        return self

    def params(self) -> Dict[str, int]:
        return {"vae": count_params(self.vae), "flow": count_params(self.flow),
                "total": count_params(self.vae) + count_params(self.flow)}

    def save(self, path, card_update: Optional[Dict] = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if card_update:
            self.card.update(card_update)
        self.card["params"] = self.params()
        payload = {"format": FORMAT, "config": json.dumps(self.cfg.as_dict()), "card": json.dumps(self.card, default=str),
                   "guidance_distilled": bool(self.guidance_distilled),
                   "vae": {k: v.detach().cpu() for k, v in self.vae.state_dict().items()},
                   "flow": {k: v.detach().cpu() for k, v in self.flow.state_dict().items()},
                   "mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}
        tmp = p.with_suffix(p.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path, device="cpu") -> "AuraNative":
        d = torch.load(str(path), map_location="cpu", weights_only=True)
        if d.get("format") != FORMAT:
            raise ValueError(f"{path} is not an Aura Native checkpoint ({d.get('format')})")
        m = cls(NativeConfig.from_dict(json.loads(d["config"])), "cpu")
        m.vae.load_state_dict(d["vae"])
        m.flow.load_state_dict(d["flow"])
        m.mean, m.std = d["mean"].float(), d["std"].float()
        m.card = json.loads(d["card"])
        m.guidance_distilled = bool(d.get("guidance_distilled", False))
        return m.to(device).eval()

    def clone(self) -> "AuraNative":
        m = AuraNative(NativeConfig.from_dict(self.cfg.as_dict()), "cpu")
        m.vae.load_state_dict({k: v.detach().cpu() for k, v in self.vae.state_dict().items()})
        m.flow.load_state_dict({k: v.detach().cpu() for k, v in self.flow.state_dict().items()})
        m.mean, m.std = self.mean.detach().cpu().clone(), self.std.detach().cpu().clone()
        m.card = json.loads(json.dumps(self.card, default=str))
        m.guidance_distilled = self.guidance_distilled
        return m.to(self.device)

    # -- conditioning tensors
    def views_to_tensors(self, views: Sequence[ViewIn]):
        fc = self.cfg.flow
        if not views:
            raise ValueError("give at least one picture")
        views = list(views)[:fc.max_views]
        S, V = fc.image_size, fc.max_views
        imgs = torch.zeros(1, V, 4, S, S)
        vp = torch.zeros(1, V, 2)
        vm = torch.zeros(1, V, dtype=torch.bool)
        for i, v in enumerate(views):
            imgs[0, i] = img_to_tensor(v.image, S)
            vp[0, i, 0], vp[0, i, 1] = np.radians(v.az), np.radians(v.el)
            vm[0, i] = True
        return imgs.to(self.device), vp.to(self.device), vm.to(self.device)

    # -- generation
    @torch.no_grad()
    def sample_z(self, views: Sequence[ViewIn], n: int = 1, steps: int = 24, cfg_scale: float = 3.0, seed: int = 0,
                 heun: bool = True) -> torch.Tensor:
        self.eval()
        imgs, vp, vm = self.views_to_tensors(views)
        return sample_latents(self.flow, imgs, vp, vm, n=n, steps=steps, cfg_scale=cfg_scale, seed=seed, heun=heun,
                              w_embed=self.guidance_distilled)

    @torch.no_grad()
    def decode_mesh(self, z_norm: torch.Tensor, res: int = 64, chunk: int = 65536, min_component: float = 0.004):
        """normalised latent grid (1,C,G,G,G) or (C,G,G,G) -> (verts, faces, colours)."""
        z = (z_norm.to(self.device) * self.std.view(1, -1, 1, 1, 1) + self.mean.view(1, -1, 1, 1, 1)) if z_norm.dim() == 5 else \
            (z_norm.to(self.device)[None] * self.std.view(1, -1, 1, 1, 1) + self.mean.view(1, -1, 1, 1, 1))
        h = self.vae.tokens(z)
        P = torch.from_numpy(G.grid_coords(res).astype(np.float32)).to(self.device)[None]
        sdf = torch.cat([self.vae.query(h, P[:, i:i + chunk])[0] for i in range(0, P.shape[1], chunk)], 1)
        grid = sdf[0].float().cpu().numpy().reshape(res, res, res)
        if not ((grid < 0).any() and (grid > 0).any()):
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3), np.float32)
        v, f = G.sdf_to_mesh(grid)
        v, f = _drop_specks(v, f, min_component)
        if len(f) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3), np.float32)
        vt = torch.from_numpy(v.astype(np.float32)).to(self.device)[None]
        c = torch.cat([self.vae.query(h, vt[:, i:i + chunk])[1] for i in range(0, vt.shape[1], chunk)], 1)[0]
        return v, f, c.float().cpu().numpy()

    def generate(self, views: Sequence[ViewIn], n: int = 1, steps: int = 24, cfg_scale: float = 3.0, seed: int = 0,
                 res: int = 64, heun: bool = True, keep_z: bool = True) -> List[Generated]:
        t0 = time.time()
        Z = self.sample_z(views, n, steps, cfg_scale, seed, heun)
        out = []
        for i in range(Z.shape[0]):
            v, f, c = self.decode_mesh(Z[i:i + 1], res)
            out.append(Generated(v, f, c, seed, Z[i].float().cpu().numpy() if keep_z else None,
                                 {"steps": steps, "cfg": cfg_scale, "res": res, "index": i, "sec": time.time() - t0}))
        return out


def _drop_specks(v: np.ndarray, f: np.ndarray, min_share: float):
    """Remove connected pieces with less than `min_share` of the faces (marching-surface dust)."""
    from ..gen3d.meshutil import component_labels

    if len(f) == 0:
        return v, f
    fl = component_labels(f, len(v))                        # connected-component id per FACE
    ids, counts = np.unique(fl, return_counts=True)
    keep_ids = ids[counts >= min_share * len(f)]
    keep = np.isin(fl, keep_ids)
    f = f[keep]
    used = np.unique(f)
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]
