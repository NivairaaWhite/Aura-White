"""Optional NVIDIA nvdiffrast backend (CUDA only). Imported lazily; any failure -> PyTorch backend."""
from __future__ import annotations

import torch

from .torch_raster import rasterize_screen

_NEAR, _FAR = 1e-3, 1e4


def _ctx(raster):
    import nvdiffrast.torch as dr

    if raster._ctx is None:
        raster._ctx = dr.RasterizeCudaContext(raster.device)
    return dr, raster._ctx


def rasterize_nvdiffrast(raster, xy, w, tris, height, width):
    """Same contract as rasterize_screen. Screen coords are converted back to clip space."""
    dr, ctx = _ctx(raster)
    x = (xy[:, 0] / width) * 2.0 - 1.0
    y = 1.0 - (xy[:, 1] / height) * 2.0          # nvdiffrast rows start at the bottom; flipped back below
    z = (w - _NEAR) / (_FAR - _NEAR) * 2.0 - 1.0
    clip = torch.stack([x * w, y * w, z * w, w], 1).contiguous().float()
    rast, _ = dr.rasterize(ctx, clip[None], tris.int().contiguous(), (height, width))
    rast = rast[0].flip(0)                        # -> image row 0 at the top
    tri = rast[..., 3].long() - 1
    u, v = rast[..., 0], rast[..., 1]
    bary = torch.stack([u, v, 1.0 - u - v], -1)
    hit = tri >= 0
    tw = w[tris.long()[tri.clamp(min=0)]]
    depth = torch.where(hit, (bary * tw).sum(-1), torch.full_like(u, float("inf")))
    bary = torch.where(hit.unsqueeze(-1), bary, torch.zeros_like(bary))
    return tri, bary, depth


class _Probe:
    def __init__(self, device):
        self.device, self._ctx = device, None


def probe(device) -> bool:
    """True if nvdiffrast works here and agrees with the PyTorch rasteriser on a small perspective scene."""
    try:
        r = _Probe(device)
        xy = torch.tensor([[10.5, 12.0], [90.0, 20.5], [40.0, 88.0], [70.0, 60.0], [15.0, 70.0]], device=device)
        w = torch.tensor([2.0, 3.5, 2.5, 4.0, 3.0], device=device)
        tris = torch.tensor([[0, 1, 2], [1, 3, 2], [0, 2, 4]], dtype=torch.long, device=device)
        a = rasterize_nvdiffrast(r, xy, w, tris, 96, 96)
        b = rasterize_screen(xy, w, tris, 96, 96)
        same_mask = ((a[0] >= 0) == (b[0] >= 0)).float().mean().item()
        both = (a[0] >= 0) & (b[0] >= 0)
        if same_mask < 0.985 or not bool(both.any()):
            return False
        depth_err = (a[2][both] - b[2][both]).abs().max().item()
        return depth_err < 5e-2
    except Exception:
        return False
