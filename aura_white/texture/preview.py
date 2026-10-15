"""Quick software renders of a textured mesh (QA sheets, turntables, tests)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..raster import Raster, interpolate, orbit_camera


def render_textured(raster: Raster, verts, faces, uv, texture: np.ndarray, cam, size: int = 384,
                    bg=(1.0, 1.0, 1.0), normals=None, light: float = 0.0):
    """Returns (rgb float32 (size,size,3), mask bool). `light` in [0,1] mixes in headlight shading."""
    dev = raster.device
    V = torch.as_tensor(verts, dtype=torch.float32, device=dev)
    Fc = torch.as_tensor(faces, dtype=torch.long, device=dev)
    out = raster.rasterize_camera(V, Fc, cam, size)
    uvi = interpolate(torch.as_tensor(uv, dtype=torch.float32, device=dev), Fc, out)
    tex = torch.as_tensor(texture, dtype=torch.float32, device=dev).permute(2, 0, 1)[None] / 255.0
    grid = (uvi * 2 - 1).unsqueeze(0)
    col = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0)
    if light > 0 and normals is not None:
        n = F.normalize(interpolate(torch.as_tensor(normals, dtype=torch.float32, device=dev), Fc, out), dim=-1)
        pos = torch.as_tensor(cam.position, dtype=torch.float32, device=dev)
        p = interpolate(V, Fc, out)
        l = F.normalize(pos - p, dim=-1)
        col = col * ((1 - light) + light * (n * l).sum(-1, keepdim=True).clamp(0.15, 1))
    m = out.mask
    bgc = torch.as_tensor(bg, dtype=torch.float32, device=dev)
    col = torch.where(m.unsqueeze(-1), col, bgc)
    return col.cpu().numpy(), m.cpu().numpy()


def turntable_sheet(raster: Raster, verts, faces, uv, texture, azimuths: Sequence[float] = (0, 60, 120, 180, 240, 300),
                    elevation: float = 12.0, size: int = 320, cam_kwargs=None, normals=None, light: float = 0.35):
    cols = []
    kw = dict(distance=5.0, fov=24.0)
    kw.update(cam_kwargs or {})
    for a in azimuths:
        img, _ = render_textured(raster, verts, faces, uv, texture, orbit_camera(a, elevation, **kw), size,
                                 normals=normals, light=light)
        cols.append(img)
    return np.concatenate(cols, axis=1)
