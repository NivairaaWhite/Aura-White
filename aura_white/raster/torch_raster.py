"""Pure-PyTorch triangle rasteriser: exact z-buffer, perspective-correct barycentrics, no extensions.

Triangles are binned by bounding-box size; every bin is evaluated as a dense (n, S, S) candidate block and
resolved with a single ``scatter_reduce(amin)`` on a packed (depth-bits, triangle-id) key - the standard
trick for a race-free z-buffer without atomics.  Cost is proportional to the covered area, so a 2M-face
mesh at 2048^2 and a 4096^2 UV atlas are both fine on a GPU, and small tests run on a CPU.
"""
from __future__ import annotations

import torch

_I64_MAX = torch.iinfo(torch.int64).max
_BARY_EPS = 1e-6


def _edge(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def rasterize_screen(xy: torch.Tensor, w: torch.Tensor, tris: torch.Tensor, height: int, width: int,
                     budget: int = 0):
    """xy: (V,2) pixel coordinates (x right, y down, pixel centres at +0.5); w: (V,) positive depth
    (perspective divisor; use ones for an affine map such as a UV atlas); tris: (F,3) int.

    Returns  tri (H,W) int64 (-1 = empty),  bary (H,W,3) float32 perspective-correct,  depth (H,W) float32
    (interpolated w, +inf where empty).  Overlaps are resolved by nearest depth, then lowest triangle id."""
    dev = xy.device
    xy = xy.float()
    w = w.float()
    tris = tris.long()
    if budget <= 0:
        budget = 8_000_000 if dev.type == "cuda" else 2_000_000

    p = xy[tris]                                 # (F,3,2)
    tw = w[tris]                                 # (F,3)
    x0, y0, x1, y1, x2, y2 = p[:, 0, 0], p[:, 0, 1], p[:, 1, 0], p[:, 1, 1], p[:, 2, 0], p[:, 2, 1]
    area = _edge(x0, y0, x1, y1, x2, y2)
    fin = torch.isfinite(p).all(dim=(1, 2)) & torch.isfinite(tw).all(1)
    ok = fin & (area.abs() > 1e-12) & (tw > 1e-6).all(1)
    minx = torch.stack([x0, x1, x2], 1).min(1).values
    maxx = torch.stack([x0, x1, x2], 1).max(1).values
    miny = torch.stack([y0, y1, y2], 1).min(1).values
    maxy = torch.stack([y0, y1, y2], 1).max(1).values
    ok &= (maxx >= 0) & (minx <= width) & (maxy >= 0) & (miny <= height)
    zero = torch.zeros_like(minx)
    ix0 = torch.where(ok, torch.ceil(minx - 0.5), zero).clamp(0, width - 1).long()
    ix1 = torch.where(ok, torch.floor(maxx - 0.5), zero - 1).clamp(-1, width - 1).long()
    iy0 = torch.where(ok, torch.ceil(miny - 0.5), zero).clamp(0, height - 1).long()
    iy1 = torch.where(ok, torch.floor(maxy - 0.5), zero - 1).clamp(-1, height - 1).long()
    bw, bh = ix1 - ix0 + 1, iy1 - iy0 + 1
    ok &= (bw > 0) & (bh > 0)
    ids = torch.nonzero(ok).squeeze(1)

    zbuf = torch.full((height * width,), _I64_MAX, dtype=torch.long, device=dev)
    if ids.numel() > 0:
        size = torch.maximum(bw[ids], bh[ids]).clamp(min=1)
        cls = torch.ceil(torch.log2(size.float())).long()
        for c in torch.unique(cls).tolist():
            grp = ids[cls == c]
            S = 1 << int(c)
            step = max(1, budget // (S * S))
            ar = torch.arange(S, device=dev)
            for s in range(0, grp.numel(), step):
                idx = grp[s:s + step]
                px = ix0[idx].view(-1, 1, 1) + ar.view(1, 1, S)
                py = iy0[idx].view(-1, 1, 1) + ar.view(1, S, 1)
                inbox = (px <= ix1[idx].view(-1, 1, 1)) & (py <= iy1[idx].view(-1, 1, 1))
                cx, cy = px.float() + 0.5, py.float() + 0.5
                a = area[idx].view(-1, 1, 1)
                X0, Y0, X1, Y1, X2, Y2 = (t[idx].view(-1, 1, 1) for t in (x0, y0, x1, y1, x2, y2))
                b0 = _edge(cx, cy, X1, Y1, X2, Y2) / a
                b1 = _edge(X0, Y0, cx, cy, X2, Y2) / a
                b2 = 1.0 - b0 - b1
                cov = inbox & (b0 >= -_BARY_EPS) & (b1 >= -_BARY_EPS) & (b2 >= -_BARY_EPS)
                if not bool(cov.any()):
                    continue
                n_i, r_i, c_i = torch.nonzero(cov, as_tuple=True)
                t_id = idx[n_i]
                bb0, bb1, bb2 = b0[n_i, r_i, c_i], b1[n_i, r_i, c_i], b2[n_i, r_i, c_i]
                tw_ = tw[t_id]
                depth = 1.0 / (bb0 / tw_[:, 0] + bb1 / tw_[:, 1] + bb2 / tw_[:, 2])
                pix = (py[n_i, r_i, 0] * width + px[n_i, 0, c_i])
                bits = depth.contiguous().view(torch.int32).long()
                key = (bits << 32) | t_id
                zbuf.scatter_reduce_(0, pix, key, reduce="amin", include_self=True)

    hit = zbuf != _I64_MAX
    tri = torch.full((height * width,), -1, dtype=torch.long, device=dev)
    bary = torch.zeros((height * width, 3), dtype=torch.float32, device=dev)
    depth = torch.full((height * width,), float("inf"), dtype=torch.float32, device=dev)
    pix = torch.nonzero(hit).squeeze(1)
    if pix.numel() > 0:
        t_id = zbuf[pix] & 0xFFFFFFFF
        tri[pix] = t_id
        cx = (pix % width).float() + 0.5
        cy = (pix // width).float() + 0.5
        a = area[t_id]
        b0 = _edge(cx, cy, x1[t_id], y1[t_id], x2[t_id], y2[t_id]) / a
        b1 = _edge(x0[t_id], y0[t_id], cx, cy, x2[t_id], y2[t_id]) / a
        b2 = 1.0 - b0 - b1
        wv = tw[t_id]
        q0, q1, q2 = b0 / wv[:, 0], b1 / wv[:, 1], b2 / wv[:, 2]
        s = q0 + q1 + q2
        bary[pix] = torch.stack([q0 / s, q1 / s, q2 / s], 1)
        depth[pix] = 1.0 / s
    return tri.view(height, width), bary.view(height, width, 3), depth.view(height, width)
