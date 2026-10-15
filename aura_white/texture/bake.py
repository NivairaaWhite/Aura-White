"""Multi-view texture baking.

Every texel of the UV atlas knows its 3D position and normal.  For each source picture (the original artwork,
optionally plus generated side views) the texel is projected into the picture, tested against a depth buffer
of the mesh for visibility and weighted by view angle and distance from the silhouette.  The original artwork
gets a large weight, so wherever it can see the surface the texture is *its* pixels (crisp, exact colours);
generated views only take over on surfaces the artwork cannot see.  Texels nothing can see are completed from
their nearest coloured neighbours in 3D."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..raster import Camera, Raster
from .fill import dilate_gutters, fill_from_neighbors


@dataclass
class ViewSource:
    image: np.ndarray                      # (h,w,3) float32 sRGB in [0,1]
    camera: Camera
    weight: float = 1.0                    # trust: the artwork ~8, generated views 1
    alpha: Optional[np.ndarray] = None     # (h,w) float [0,1] foreground of the picture (artwork cut-out)
    name: str = ""
    harmonize: bool = False                # match colour statistics to the reference view first


@dataclass
class BakeResult:
    texture: np.ndarray                    # (T,T,3) uint8
    occupied: np.ndarray                   # (T,T) bool: texel belongs to a chart
    seen: np.ndarray                       # (T,T) float: 1 = seen by a picture, 0 = filled in
    usage: Dict[str, float] = field(default_factory=dict)   # share of chart texels dominated by each view
    seconds: float = 0.0


class _ViewPrep:
    def __init__(self, raster: Raster, verts, faces, view: ViewSource, max_res: int, edge_frac: float):
        dev = raster.device
        img = torch.as_tensor(np.ascontiguousarray(view.image), dtype=torch.float32, device=dev)
        self.img = img.permute(2, 0, 1)[None]
        self.cam, self.weight = view.camera, view.weight
        dr = int(min(max(img.shape[0], 512), max_res))
        self.dr = dr
        out = raster.rasterize_camera(verts, faces, view.camera, dr)
        depth = torch.where(torch.isfinite(out.depth), out.depth, torch.full_like(out.depth, 1e6))
        self.dmin = -F.max_pool2d((-depth)[None, None], 3, 1, 1)[0, 0]      # conservative 3x3 nearest depth
        fg = out.mask.float()
        if view.alpha is not None:
            a = torch.as_tensor(view.alpha, dtype=torch.float32, device=dev)[None, None]
            a = F.interpolate(a, size=(dr, dr), mode="area")[0, 0]
            fg = fg * (a > 0.5).float()
        k = max(2, int(round(edge_frac * dr)))
        cur, acc = fg[None, None], torch.zeros_like(fg)[None, None]
        for _ in range(k):
            cur = -F.max_pool2d(-cur, 3, 1, 1)
            acc = acc + cur
        self.edge = acc / k
        self.pos = torch.as_tensor(view.camera.position, dtype=torch.float32, device=dev)
        self.m = torch.as_tensor(view.camera.w2c, dtype=torch.float32, device=dev)
        self.coverage = fg.mean().item()

    def sample(self, P: torch.Tensor, N: torch.Tensor, exponent: float):
        cam, dr = self.cam, self.dr
        pc = P @ self.m[:3, :3].T + self.m[:3, 3]
        w = -pc[:, 2]
        ws = w.clamp(min=1e-6)
        gx = cam.f * pc[:, 0] / ws + cam.cx
        gy = -(cam.f * pc[:, 1] / ws + cam.cy)
        inb = (gx.abs() < 1.0) & (gy.abs() < 1.0) & (w > 1e-3)
        ix = ((gx * 0.5 + 0.5) * dr).long().clamp(0, dr - 1)
        iy = ((gy * 0.5 + 0.5) * dr).long().clamp(0, dr - 1)
        to_cam = self.pos - P
        cosv = ((N * to_cam).sum(-1) / to_cam.norm(dim=-1).clamp(min=1e-9)).clamp(0.0, 1.0)
        tan = (1 - cosv ** 2).clamp(min=0).sqrt() / cosv.clamp(min=1e-3)
        foot = 2.0 * w / (cam.f * dr)
        vis = inb & (w <= self.dmin[iy, ix] + foot * (1.5 + 3.0 * tan.clamp(max=4.0)))
        grid = torch.stack([gx, gy], -1).view(1, 1, -1, 2)
        edge = F.grid_sample(self.edge, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0, 0]
        col = F.grid_sample(self.img, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
        wgt = self.weight * cosv.pow(exponent) * edge * vis.float()
        return col, wgt


def _harmonize_params(ref: _ViewPrep, view: _ViewPrep, P, N, exponent, stride=7):
    """Per-channel gain/bias mapping `view` colours onto `ref` colours over their overlap."""
    Ps, Ns = P[::stride], N[::stride]
    cr, wr = ref.sample(Ps, Ns, 1.0)
    cv, wv = view.sample(Ps, Ns, 1.0)
    ok = (wr > 0.3 * ref.weight) & (wv > 0.3 * view.weight)
    if int(ok.sum()) < 400:
        return None
    x, y = cv[ok], cr[ok]
    mx, my = x.mean(0), y.mean(0)
    var = ((x - mx) ** 2).mean(0).clamp(min=1e-6)
    cov = ((x - mx) * (y - my)).mean(0)
    gain = (cov / var).clamp(0.75, 1.33)
    bias = (my - gain * mx).clamp(-0.12, 0.12)
    return gain, bias


def bake_texture(raster: Raster, verts: np.ndarray, faces: np.ndarray, uv: np.ndarray, normals: np.ndarray,
                 views: List[ViewSource], size: int = 2048, exponent: float = 4.0, edge_frac: float = 0.012,
                 min_weight: float = 0.02, max_view_res: int = 2048, chunk: int = 1_500_000,
                 gutter: Optional[int] = None, progress=None) -> BakeResult:
    t0 = time.time()
    dev = raster.device
    V = torch.as_tensor(verts, dtype=torch.float32, device=dev)
    Fc = torch.as_tensor(faces, dtype=torch.long, device=dev)
    Nv = torch.as_tensor(normals, dtype=torch.float32, device=dev)
    UV = torch.as_tensor(uv, dtype=torch.float32, device=dev)
    say = progress or (lambda *_: None)

    # 1. where every texel lives in 3D
    ur = raster.rasterize_uv(UV, Fc, size)
    flat_tri = ur.tri.reshape(-1)
    valid = torch.nonzero(flat_tri >= 0).squeeze(1)
    n = len(valid)
    if n == 0:
        raise RuntimeError("the UV atlas covers no texels")
    tid = flat_tri[valid]
    bary = ur.bary.reshape(-1, 3)[valid]
    P = torch.empty((n, 3), device=dev)
    Nn = torch.empty((n, 3), device=dev)
    for s in range(0, n, chunk):
        f = Fc[tid[s:s + chunk]]
        b = bary[s:s + chunk].unsqueeze(-1)
        P[s:s + chunk] = (V[f] * b).sum(1)
        Nn[s:s + chunk] = F.normalize((Nv[f] * b).sum(1), dim=-1)
    del bary, tid, ur

    # 2. project every picture
    preps = []
    for i, v in enumerate(views):
        say(f"preparing view {i + 1}/{len(views)}: {v.name}")
        preps.append(_ViewPrep(raster, V, Fc, v, max_view_res, edge_frac))
    ref = max(range(len(views)), key=lambda i: views[i].weight)
    acc = torch.zeros((n, 3), device=dev)
    wsum = torch.zeros(n, device=dev)
    wbest = torch.zeros(n, device=dev)
    dom = torch.full((n,), -1, dtype=torch.int16, device=dev)
    for i, (v, pv) in enumerate(zip(views, preps)):
        say(f"projecting {v.name or i}")
        gb = None
        if v.harmonize and i != ref:
            gb = _harmonize_params(preps[ref], pv, P, Nn, exponent)
        for s in range(0, n, chunk):
            col, w = pv.sample(P[s:s + chunk], Nn[s:s + chunk], exponent)
            if gb is not None:
                col = (col * gb[0] + gb[1]).clamp(0, 1)
            acc[s:s + chunk] += col * w.unsqueeze(-1)
            wsum[s:s + chunk] += w
            better = w > wbest[s:s + chunk]
            wbest[s:s + chunk] = torch.where(better, w, wbest[s:s + chunk])
            dom[s:s + chunk][better] = i

    # 3. blend, then complete what nothing could see (soft transition, computed in 3D)
    color = acc / wsum.clamp(min=1e-9).unsqueeze(-1)
    conf = (wsum / (3.0 * min_weight)).clamp(0, 1)
    say("filling unseen texels")
    filled = fill_from_neighbors(P, color, wsum, 3.0 * min_weight)
    a = (conf * conf * (3 - 2 * conf)).unsqueeze(-1)
    color = a * color + (1 - a) * filled
    color = torch.where((wsum <= 1e-9).unsqueeze(-1), filled, color)

    tex = torch.zeros(size * size, 3, device=dev)
    tex[valid] = color
    occ = torch.zeros(size * size, dtype=torch.bool, device=dev)
    occ[valid] = True
    seen = torch.zeros(size * size, device=dev)
    seen[valid] = conf
    usage = {}
    for i, v in enumerate(views):
        usage[v.name or f"view{i}"] = float((dom == i).float().mean().item())
    tex = dilate_gutters(tex.view(size, size, 3), occ.view(size, size), gutter or max(6, size // 128))
    out = (tex.clamp(0, 1) * 255.0 + 0.5).byte().cpu().numpy()
    return BakeResult(out, occ.view(size, size).cpu().numpy(), seen.view(size, size).cpu().numpy(), usage,
                      time.time() - t0)
