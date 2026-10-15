"""High-level API:  AuraWhite.from_pretrained().generate("photo.png").save("out.glb")"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from ._compat import load_state_dict_file, pick_device, pick_dtype
from .config import AuraConfig
from .image import Prepared, prepare_image
from .mesh import keep_large_components, save_mesh, signed_volume, surface_nets, taubin_smooth
from .model import AuraNet
from .volume import build_volume, grid_to_world
from .weights import normalize_state_dict, resolve_weights

Progress = Optional[Callable[[str, float], None]]


class EmptyMeshError(RuntimeError):
    """The density field has no surface at this threshold."""


def orient_vertices(verts: np.ndarray, up: str = "y", yaw: float = 0.0) -> np.ndarray:
    """Model frame (z up, input photo seen from +x) -> a standard frame.

    up="y": glTF/OBJ convention - Y up, and the side the photo was taken from faces +Z.
    up="z": unchanged (what TripoSR writes).  yaw rotates about the vertical axis."""
    v = np.asarray(verts, dtype=np.float32)
    if up == "z":
        out = v.copy()
        if yaw:
            t = np.radians(yaw)
            c, s = np.cos(t), np.sin(t)
            out = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)
        return out.astype(np.float32)
    out = np.stack([v[:, 1], v[:, 2], v[:, 0]], axis=1)  # cyclic permutation: keeps handedness
    if yaw:
        t = np.radians(yaw)
        c, s = np.cos(t), np.sin(t)
        out = np.stack([c * out[:, 0] + s * out[:, 2], out[:, 1], -s * out[:, 0] + c * out[:, 2]], axis=1)
    return out.astype(np.float32)


@dataclass
class MeshResult:
    vertices: np.ndarray                 # (V,3) float32, model frame
    faces: np.ndarray                    # (F,3) int32, counter-clockwise from outside
    colors: np.ndarray                   # (V,3) uint8 sRGB
    timings: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, object] = field(default_factory=dict)
    prepared: Optional[Prepared] = None
    scene_code: Optional[torch.Tensor] = field(default=None, repr=False)

    def oriented(self, up: str = "y", yaw: float = 0.0) -> np.ndarray:
        return orient_vertices(self.vertices, up, yaw)

    def save(self, path, fmt: Optional[str] = None, up: str = "y", yaw: float = 0.0) -> Path:
        return save_mesh(path, self.oriented(up, yaw), self.faces, self.colors, None, fmt)

    def save_all(self, directory, stem: str = "mesh", formats: Sequence[str] = ("glb", "obj"),
                 up: str = "y", yaw: float = 0.0) -> Dict[str, Path]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        v = self.oriented(up, yaw)
        return {f: save_mesh(d / f"{stem}.{f}", v, self.faces, self.colors, None, f) for f in formats}


class AuraWhite:
    def __init__(self, net: AuraNet, device: torch.device, dtype: torch.dtype, weights_path: Optional[Path] = None):
        self.net = net.eval()
        self.device = device
        self.dtype = dtype
        self.cfg = net.cfg
        self.weights_path = weights_path
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(cls, weights: Optional[str] = None, device: str = "auto", precision: str = "auto",
                        allow_download: bool = True, trust_pickle: bool = False, progress: Progress = None,
                        cfg: Optional[AuraConfig] = None) -> "AuraWhite":
        dev = pick_device(device)
        dtype = pick_dtype(dev, precision)
        if dev.type == "cpu" and dtype == torch.float16:
            dtype = torch.float32  # CPU half-precision kernels are missing/slow; never worth it
        path = resolve_weights(weights, allow_download, progress, trust_pickle)
        if progress:
            progress("loading weights", 0.0)
        cfg = cfg or AuraConfig()
        net = AuraNet(cfg)
        sd = normalize_state_dict(load_state_dict_file(path, trust_pickle))
        try:
            missing, unexpected = net.load_state_dict(sd, strict=False)
        except RuntimeError as e:
            raise RuntimeError(f"{Path(path).name} does not match the Aura White / TripoSR architecture: {e}") from e
        if missing or unexpected:
            raise RuntimeError(
                f"{Path(path).name} does not match the expected architecture. "
                f"missing={list(missing)[:4]}{'...' if len(missing) > 4 else ''} "
                f"unexpected={list(unexpected)[:4]}{'...' if len(unexpected) > 4 else ''}"
            )
        del sd
        net.to(dev).set_precision(dtype)
        if progress:
            progress("loading weights", 1.0)
        return cls(net, dev, dtype, Path(path))

    @classmethod
    def from_random(cls, cfg: Optional[AuraConfig] = None, device: str = "cpu", seed: int = 0) -> "AuraWhite":
        """Randomly initialised model (tiny by default): for self-tests, no weights needed."""
        torch.manual_seed(seed)
        cfg = cfg or AuraConfig.tiny()
        dev = pick_device(device)
        net = AuraNet(cfg).to(dev).set_precision(torch.float32)
        return cls(net, dev, torch.float32, None)

    def info(self) -> dict:
        return {"device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "parameters_m": round(sum(p.numel() for p in self.net.parameters()) / 1e6, 1),
                "weights": str(self.weights_path) if self.weights_path else "random (self-test)"}

    # ------------------------------------------------------------------ steps
    def encode_image(self, image) -> torch.Tensor:
        """image: PIL image or float32 (H,W,3) array in [0,1]."""
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            arr = np.ascontiguousarray(image, dtype=np.float32)
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(self.device)
        return self.net.encode(x)[0]

    def auto_threshold(self, code: torch.Tensor, n: int = 20000) -> float:
        """Self-test helper for random models: a level between the median and the 90th percentile."""
        g = torch.Generator().manual_seed(1)
        pts = ((torch.rand(n, 3, generator=g) * 2 - 1) * self.cfg.radius).to(self.device)
        d, _ = self.net.query(code, pts)
        med, hi = torch.quantile(d.float(), 0.5).item(), torch.quantile(d.float(), 0.9).item()
        return float((max(med, 1e-9) * max(hi, 1e-9)) ** 0.5)

    def _density_fn(self, code: torch.Tensor, query_chunk: int):
        def fn(points: np.ndarray) -> np.ndarray:
            p = torch.from_numpy(points).to(self.device)
            d, _ = self.net.query(code, p, query_chunk)
            return d.float().cpu().numpy()
        return fn

    def _colors(self, code: torch.Tensor, world: np.ndarray, query_chunk: int) -> np.ndarray:
        out = []
        for s in range(0, len(world), query_chunk):
            p = torch.from_numpy(np.ascontiguousarray(world[s : s + query_chunk])).to(self.device)
            _, c = self.net.query(code, p, query_chunk)
            out.append((c.float().clamp(0, 1) * 255.0).round().byte().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 3), np.uint8)

    # ------------------------------------------------------------------ main API
    def generate(self, image, *, resolution: int = 256, threshold="25", remove_bg: bool = True,
                 foreground_ratio: float = 0.85, matting: str = "auto", smooth: int = 2,
                 min_part_ratio: float = 0.02, adaptive: bool = True, low_memory: bool = False,
                 progress: Progress = None) -> MeshResult:
        """Photo -> coloured triangle mesh. `image`: path, bytes, PIL image, array or a Prepared.
        `threshold`: density level of the surface (25 = TripoSR default) or "auto" (self-test models).
        Thread-safe (calls are serialised)."""
        with self._lock:
            return self._generate(image, resolution, threshold, remove_bg, foreground_ratio, matting, smooth,
                                  min_part_ratio, adaptive, low_memory, progress)

    def _generate(self, image, resolution, threshold, remove_bg, fg_ratio, matting, smooth, min_part, adaptive,
                  low_memory, progress) -> MeshResult:
        prog = progress or (lambda *_: None)
        resolution = int(np.clip(resolution, 32, 1024))
        chunk = 32768 if low_memory else 262144
        qchunk = 16384 if low_memory else 131072
        T: Dict[str, float] = {}
        t_all = time.time()

        t = time.time()
        prog("preparing image", 0.0)
        prepared = image if isinstance(image, Prepared) else prepare_image(image, remove_bg, fg_ratio, matting)
        T["preprocess"] = time.time() - t

        t = time.time()
        prog("encoding image", 0.0)
        code = self.encode_image(prepared.rgb)
        T["encode"] = time.time() - t

        if isinstance(threshold, str):
            if threshold.strip().lower() == "auto":
                thr = self.auto_threshold(code)
            else:
                try:
                    thr = float(threshold)
                except ValueError:
                    raise ValueError("threshold must be a number or 'auto'") from None
        else:
            thr = float(threshold)

        t = time.time()
        vol, vstats = build_volume(self._density_fn(code, qchunk), self.cfg.radius, resolution, thr,
                                   adaptive=adaptive, chunk=chunk, progress=prog)
        T["density"] = time.time() - t

        t = time.time()
        prog("extracting surface", 0.0)
        pad = np.pad(vol, 1, mode="constant", constant_values=0.0)  # closes shapes that touch the box
        del vol
        v, f = surface_nets(pad, thr)
        del pad
        if f.shape[0] == 0:
            raise EmptyMeshError("No surface found. Lower --threshold, or use a clearer photo of a single object.")
        world = grid_to_world(v, resolution, self.cfg.radius, pad=1)
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
        f = f[ok]
        world, f, _ = keep_large_components(world, f, None, min_part)
        world = taubin_smooth(world, f, smooth)
        if signed_volume(world, f) < 0:  # safety net; construction already guarantees outward faces
            f = f[:, ::-1].copy()
        T["mesh"] = time.time() - t

        t = time.time()
        prog("colouring vertices", 0.0)
        colors = self._colors(code, world, qchunk)
        T["colors"] = time.time() - t
        T["total"] = time.time() - t_all
        prog("done", 1.0)

        stats = dict(vstats, vertices=int(len(world)), faces=int(len(f)), resolution=resolution, threshold=thr,
                     query_fraction=float(vstats["fraction"]), matting=prepared.matting,
                     warnings=list(prepared.warnings))
        return MeshResult(world, f.astype(np.int32), colors, T, stats, prepared, code)

    # ------------------------------------------------------------------ preview
    def render_turntable(self, result: MeshResult, n_views: int = 16, size: int = 192) -> List[np.ndarray]:
        """Volume-rendered turntable frames (slow on CPU; optional)."""
        if result.scene_code is None:
            raise ValueError("this result has no scene code")
        return self.net.render(result.scene_code, n_views=n_views, size=size)


def _turntable(self, result: "MeshResult", path, n_views: int = 16, size: int = 192, fps: int = 12) -> Path:
    return save_gif(self.render_turntable(result, n_views, size), path, fps)


AuraWhite.turntable = _turntable


def save_gif(frames: Sequence[np.ndarray], path, fps: int = 12) -> Path:
    ims = [Image.fromarray(f) for f in frames]
    path = Path(path)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=False)
    return path
