"""Backend-independent rasteriser front end (pure PyTorch or nvdiffrast)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from .camera import Camera
from .torch_raster import rasterize_screen


@dataclass
class RasterOut:
    tri: torch.Tensor      # (H,W) int64, -1 where empty
    bary: torch.Tensor     # (H,W,3) float32, perspective-correct barycentrics
    depth: torch.Tensor    # (H,W) float32, view-space depth (inf where empty)

    @property
    def mask(self) -> torch.Tensor:
        return self.tri >= 0


def interpolate(attr: torch.Tensor, tris: torch.Tensor, out: RasterOut) -> torch.Tensor:
    """attr: (V,C) per-vertex attribute -> (H,W,C), zeros where empty."""
    h, w = out.tri.shape
    res = torch.zeros((h, w, attr.shape[1]), dtype=attr.dtype, device=attr.device)
    m = out.tri >= 0
    if bool(m.any()):
        tid = out.tri[m]
        a = attr[tris.long()[tid]]                          # (n,3,C)
        res[m] = (a * out.bary[m].unsqueeze(-1).to(attr.dtype)).sum(1)
    return res


def camera_to_screen(verts: torch.Tensor, cam: Camera, height: int, width: int):
    """(V,3) world points -> pixel coords (V,2) and depth w (V,). Depth is distance along the view axis."""
    m = torch.as_tensor(cam.w2c, dtype=torch.float32, device=verts.device)
    pc = verts.float() @ m[:3, :3].T + m[:3, 3]
    w = -pc[:, 2]
    ws = torch.where(w.abs() < 1e-9, torch.full_like(w, 1e-9), w)
    aspect = width / height
    nx = cam.f * pc[:, 0] / ws / aspect + cam.cx
    ny = cam.f * pc[:, 1] / ws + cam.cy
    sx = (nx * 0.5 + 0.5) * width
    sy = (0.5 - ny * 0.5) * height
    return torch.stack([sx, sy], 1), w


class Raster:
    """rasterize_camera / rasterize_uv with a common result type."""

    def __init__(self, device: torch.device, backend: str = "torch"):
        self.device = torch.device(device)
        self.backend = backend
        self._ctx = None

    def rasterize_camera(self, verts: torch.Tensor, tris: torch.Tensor, cam: Camera, height: int,
                         width: Optional[int] = None) -> RasterOut:
        width = width or height
        xy, w = camera_to_screen(verts.to(self.device), cam, height, width)
        return self._run(xy, w, tris.to(self.device), height, width)

    def rasterize_uv(self, uv: torch.Tensor, tris: torch.Tensor, size: int) -> RasterOut:
        uv = uv.to(self.device).float()
        xy = torch.stack([uv[:, 0] * size, uv[:, 1] * size], 1)   # v grows downwards = image rows
        return self._run(xy, torch.ones(len(uv), device=self.device), tris.to(self.device), size, size)

    def _run(self, xy, w, tris, height, width) -> RasterOut:
        if self.backend == "nvdiffrast":
            from .nvdiff import rasterize_nvdiffrast

            return RasterOut(*rasterize_nvdiffrast(self, xy, w, tris, height, width))
        return RasterOut(*rasterize_screen(xy, w, tris, height, width))


_cache = {}


def get_raster(device="auto", backend: str = "auto") -> Raster:
    """backend: auto | torch | nvdiffrast. `auto` picks nvdiffrast only if it imports, creates a CUDA
    context and reproduces the PyTorch rasteriser on a probe scene; otherwise pure PyTorch.
    Environment override: AURA_WHITE_RASTER=torch|nvdiffrast."""
    backend = os.environ.get("AURA_WHITE_RASTER", backend).lower()
    if isinstance(device, str):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
    key = (str(device), backend)
    if key in _cache:
        return _cache[key]
    chosen = "torch"
    if backend in ("auto", "nvdiffrast") and device.type == "cuda":
        try:
            from .nvdiff import probe

            if probe(device):
                chosen = "nvdiffrast"
            elif backend == "nvdiffrast":
                raise RuntimeError("nvdiffrast failed its self-test")
        except Exception:
            if backend == "nvdiffrast":
                raise
    elif backend == "nvdiffrast":
        raise RuntimeError("nvdiffrast needs a CUDA device")
    r = Raster(device, chosen)
    _cache[key] = r
    return r
