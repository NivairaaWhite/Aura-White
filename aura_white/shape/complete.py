"""Inference API: complete the hidden side of a single-view mesh with an Aura Shape checkpoint."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .model import FlowMatching, ShapeDiT
from . import sdf as S


class ShapeCompleter:
    def __init__(self, model: ShapeDiT, device: str = "auto"):
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else
                                   ("cpu" if device == "auto" else device))
        self.model = model.to(self.device).eval()
        self.flow = FlowMatching()
        self.res = model.res

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "ShapeCompleter":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = ShapeDiT(**ck["config"])
        model.load_state_dict(ck.get("ema") or ck["model"])
        return cls(model, device)

    @torch.no_grad()
    def complete_tsdf(self, tsdf: np.ndarray, mask: np.ndarray, steps: int = 32, seed: int = 0) -> np.ndarray:
        cond = torch.from_numpy(np.where(mask, tsdf, 0.0).astype(np.float32))[None, None].to(self.device)
        m = torch.from_numpy(mask.astype(np.float32))[None, None].to(self.device)
        out = self.flow.sample(self.model, cond, m, steps=steps, seed=seed)
        return out[0, 0].cpu().numpy()

    def complete_mesh(self, verts: np.ndarray, faces: np.ndarray, steps: int = 32, seed: int = 0,
                      fit: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """`verts` in the canonical frame (viewer at +x).  The mesh is fitted into [-fit, fit]^3, voxelised, the part
        a camera at +x can see is kept as the observation and the rest is generated."""
        v = np.asarray(verts, np.float64)
        c = 0.5 * (v.min(0) + v.max(0))
        s = fit / max(float(np.abs(v - c).max()), 1e-9)
        sdf = S.mesh_to_sdf((v - c) * s, faces, self.res)
        obs = S.observed_mask(sdf)
        out = self.complete_tsdf(S.to_tsdf(sdf, self.res), obs, steps, seed)
        vv, ff = S.sdf_to_mesh(out * (S.TRUNC * 2.0 / self.res))
        return (vv / s + c).astype(np.float32), ff
