"""Triplane query + differentiable-style volume renderer (used for previews only; the mesh is
extracted from the density field directly)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .._compat import cross
from ..config import AuraConfig

_EXP_CLAMP = 30.0  # exp(30) ~ 1e13: finite in fp32, never reached by real densities


def query_triplane(cfg: AuraConfig, decoder, triplane: torch.Tensor, positions: torch.Tensor, chunk: int = 131072,
                   want_color: bool = True):
    """triplane: (3, C, H, W) fp32.  positions: (N, 3) in world units (-radius..radius).
    Returns density_act (N,) and color (N, 3) in [0, 1]."""
    r = cfg.radius
    dens, cols = [], []
    for s in range(0, positions.shape[0], chunk):
        p = positions[s : s + chunk].to(triplane.dtype) / r  # -> (-1, 1)
        grid = torch.stack((p[:, [0, 1]], p[:, [0, 2]], p[:, [1, 2]]), dim=0)  # (3, n, 2)
        feats = F.grid_sample(triplane, grid.unsqueeze(1), mode="bilinear", align_corners=False)  # (3,C,1,n)
        feats = feats.squeeze(2).permute(2, 0, 1).reshape(p.shape[0], -1)  # (n, 3C)
        d, c = decoder(feats)
        dens.append(torch.exp((d[:, 0] + cfg.density_bias).clamp(max=_EXP_CLAMP)))
        if want_color:
            cols.append(torch.sigmoid(c))
    return torch.cat(dens, 0), (torch.cat(cols, 0) if want_color else None)


def _ray_box(o: torch.Tensor, d: torch.Tensor, radius: float, thresh: float = 0.01):
    d = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
    lim = (1.0 - 1e-3) * radius
    t0 = (lim - o) / d
    t1 = (-lim - o) / d
    near = torch.minimum(t0, t1).amax(-1).clamp_min(0.0)
    far = torch.maximum(t0, t1).amin(-1)
    valid = (far - near) > thresh
    return torch.where(valid, near, torch.zeros_like(near)), torch.where(valid, far, torch.zeros_like(far)), valid


def render_rays(cfg: AuraConfig, decoder, triplane, rays_o, rays_d, ray_chunk: int = 4096):
    """Front-to-back alpha compositing over a white background. rays_*: (R, 3) -> (R, 3)."""
    out = []
    n_s = cfg.samples_per_ray
    for s in range(0, rays_o.shape[0], ray_chunk):
        o, d = rays_o[s : s + ray_chunk], rays_d[s : s + ray_chunk]
        n = o.shape[0]
        near, far, valid = _ray_box(o, d, cfg.radius)
        rgb = torch.zeros(n, 3, dtype=torch.float32, device=o.device)
        opacity = torch.zeros(n, dtype=torch.float32, device=o.device)
        if valid.any():
            ov, dv, nv, fv = o[valid], d[valid], near[valid], far[valid]
            t_edges = torch.linspace(0, 1, n_s + 1, device=o.device)
            t_mid = (t_edges[:-1] + t_edges[1:]) / 2.0
            z = nv[:, None] * (1 - t_mid[None]) + fv[:, None] * t_mid[None]
            xyz = ov[:, None, :] + z[..., None] * dv[:, None, :]
            dens, col = query_triplane(cfg, decoder, triplane, xyz.reshape(-1, 3))
            dens, col = dens.view(-1, n_s), col.view(-1, n_s, 3)
            delta = (t_edges[1:] - t_edges[:-1])[None]
            alpha = 1 - torch.exp(-delta * dens)
            trans = torch.cat([torch.ones_like(alpha[:, :1]), torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)], dim=-1)
            w = alpha * trans
            rgb[valid] = (w[..., None] * col).sum(-2)
            opacity[valid] = w.sum(-1)
        out.append(rgb + (1 - opacity)[:, None])
    return torch.cat(out, 0)


def spherical_cameras(n_views: int, elevation_deg: float, distance: float, fovy_deg: float, height: int, width: int):
    """Turntable cameras. Same convention as TripoSR (z up, azimuth 0 on +x)."""
    az = torch.linspace(0, 360.0, n_views + 1)[:n_views] * math.pi / 180
    el = torch.full_like(az, elevation_deg * math.pi / 180)
    pos = torch.stack(
        [distance * torch.cos(el) * torch.cos(az), distance * torch.cos(el) * torch.sin(az), distance * torch.sin(el)], -1
    )
    look = F.normalize(-pos, dim=-1)
    up0 = torch.tensor([0.0, 0.0, 1.0]).expand(n_views, 3)
    right = F.normalize(cross(look, up0), dim=-1)
    up = F.normalize(cross(right, look), dim=-1)
    rot = torch.stack([right, up, -look], dim=-1)  # (V, 3, 3) camera -> world

    focal = 0.5 * height / math.tan(0.5 * math.radians(fovy_deg))
    xs = (torch.arange(width, dtype=torch.float32) + 0.5 - width / 2) / focal
    ys = -(torch.arange(height, dtype=torch.float32) + 0.5 - height / 2) / focal
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    dirs = F.normalize(torch.stack([gx, gy, -torch.ones_like(gx)], -1), dim=-1)  # (H, W, 3)
    rays_d = torch.einsum("vij,hwj->vhwi", rot, dirs)
    rays_o = pos[:, None, None, :].expand_as(rays_d)
    return rays_o, rays_d
