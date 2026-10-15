"""Filling texels that no picture could see, and padding the gutters between UV charts."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fill_from_neighbors(points: torch.Tensor, color: torch.Tensor, confidence: torch.Tensor, threshold: float,
                        k: int = 8, max_sources: int = 250_000) -> torch.Tensor:
    """Give every low-confidence texel the inverse-distance-weighted colour of its k nearest confident texels
    *in 3D* (so the result is seamless across UV islands, unlike filling in texture space).
    points (N,3), color (N,3), confidence (N,). Returns the completed colours (N,3)."""
    src = torch.nonzero(confidence >= threshold).squeeze(1)
    qry = torch.nonzero(confidence < threshold).squeeze(1)
    out = color.clone()
    if len(qry) == 0:
        return out
    if len(src) == 0:
        out[qry] = 0.5
        return out
    kk = min(k, len(src))
    try:
        from scipy.spatial import cKDTree

        sp = points[src].cpu().numpy()
        if len(src) > max_sources * 2:
            sub = np.random.default_rng(0).choice(len(src), max_sources * 2, replace=False)
            src = src[torch.as_tensor(sub, device=src.device)]
            sp = points[src].cpu().numpy()
        tree = cKDTree(sp)
        dist, idx = tree.query(points[qry].cpu().numpy(), k=kk, workers=-1)
        dist = torch.as_tensor(dist.reshape(len(qry), kk), dtype=torch.float32)
        idx = torch.as_tensor(idx.reshape(len(qry), kk), dtype=torch.long)
        wgt = 1.0 / (dist ** 2 + 1e-8)
        cs = color[src].cpu()
        out[qry] = ((cs[idx] * wgt.unsqueeze(-1)).sum(1) / wgt.sum(1, keepdim=True)).to(out.device)
        return out
    except ImportError:
        pass
    # scipy-free path: chunked brute force over a bounded set of sources
    if len(src) > 60_000:
        g = torch.Generator().manual_seed(0)
        src = src[torch.randperm(len(src), generator=g)[:60_000].to(src.device)]
    sp, cs = points[src], color[src]
    step = max(256, int(2.5e7 // max(len(src), 1)))
    for s in range(0, len(qry), step):
        q = qry[s:s + step]
        d = torch.cdist(points[q], sp)
        dist, idx = torch.topk(d, kk, dim=1, largest=False)
        wgt = 1.0 / (dist ** 2 + 1e-8)
        out[q] = (cs[idx] * wgt.unsqueeze(-1)).sum(1) / wgt.sum(1, keepdim=True)
    return out


def dilate_gutters(tex: torch.Tensor, occupied: torch.Tensor, iterations: int) -> torch.Tensor:
    """Extend chart colours into the empty texels around them (needed so bilinear filtering and mip-mapping
    never mix in black).  tex (H,W,3); occupied (H,W) bool."""
    img = tex.permute(2, 0, 1)[None].clone()
    occ = occupied.float()[None, None]
    kern = torch.ones(1, 1, 3, 3, device=tex.device)
    for _ in range(iterations):
        den = F.conv2d(occ, kern, padding=1)
        num = F.conv2d(img * occ, kern.expand(3, 1, 3, 3).contiguous(), padding=1, groups=3)
        grow = (occ < 0.5) & (den > 0)
        if not bool(grow.any()):
            break
        img = torch.where(grow.expand_as(img), num / den.clamp(min=1e-6), img)
        occ = torch.where(grow, torch.ones_like(occ), occ)
    return img[0].permute(1, 2, 0)
