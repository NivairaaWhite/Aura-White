"""A small exact z-buffer rasteriser in NumPy, camera helpers and a preview renderer.

Used by the forge so that residual completion, silhouette snapping and QA renders work without PyTorch.  It follows
the same conventions as `aura_white.raster` (OpenGL-style camera looking down -z, `f` in NDC units, principal point
offset (cx, cy), pixel centres at +0.5, y down) so results interchange with the torch rasteriser."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------------------------------- camera
def project(cam, pts: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """World points (N,3) -> pixel coordinates (N,2) and view depth w (N,).  `cam` needs w2c, f, cx, cy."""
    m = np.asarray(cam.w2c, np.float64)
    pc = np.asarray(pts, np.float64) @ m[:3, :3].T + m[:3, 3]
    w = -pc[:, 2]
    ws = np.where(np.abs(w) < 1e-9, 1e-9, w)
    aspect = width / height
    nx = cam.f * pc[:, 0] / ws / aspect + cam.cx
    ny = cam.f * pc[:, 1] / ws + cam.cy
    return np.stack([(nx * 0.5 + 0.5) * width, (0.5 - ny * 0.5) * height], 1), w


def unproject(cam, sx: np.ndarray, sy: np.ndarray, w: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pixel coordinates + view depth -> world points."""
    aspect = width / height
    nx = (np.asarray(sx, np.float64) / width - 0.5) * 2.0
    ny = (0.5 - np.asarray(sy, np.float64) / height) * 2.0
    w = np.asarray(w, np.float64)
    pc = np.stack([(nx - cam.cx) * w * aspect / cam.f, (ny - cam.cy) * w / cam.f, -w], 1)
    c2w = np.linalg.inv(np.asarray(cam.w2c, np.float64))
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def pixel_size(cam, w: float, height: int) -> float:
    """World length of one pixel at view depth `w` (vertical direction)."""
    return 2.0 * w / (cam.f * height)


# ---------------------------------------------------------------------------------------------------- raster
def _edge(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def rasterize(xy: np.ndarray, w: np.ndarray, tris: np.ndarray, height: int, width: int, budget: int = 1_500_000):
    """Exact z-buffer.  Returns (tri (H,W) int64 with -1 = empty, bary (H,W,3) float32 perspective-correct,
    depth (H,W) float32 interpolated w with +inf = empty)."""
    xy = np.asarray(xy, np.float64)
    w = np.asarray(w, np.float64)
    tris = np.asarray(tris, np.int64)
    p = xy[tris]
    tw = w[tris]
    x0, y0, x1, y1, x2, y2 = p[:, 0, 0], p[:, 0, 1], p[:, 1, 0], p[:, 1, 1], p[:, 2, 0], p[:, 2, 1]
    area = _edge(x0, y0, x1, y1, x2, y2)
    ok = np.isfinite(p).all((1, 2)) & np.isfinite(tw).all(1) & (np.abs(area) > 1e-12) & (tw > 1e-9).all(1)
    minx, maxx = p[:, :, 0].min(1), p[:, :, 0].max(1)
    miny, maxy = p[:, :, 1].min(1), p[:, :, 1].max(1)
    ok &= (maxx >= 0) & (minx <= width) & (maxy >= 0) & (miny <= height)
    ix0 = np.clip(np.ceil(minx - 0.5), 0, width - 1).astype(np.int64)
    ix1 = np.clip(np.floor(maxx - 0.5), -1, width - 1).astype(np.int64)
    iy0 = np.clip(np.ceil(miny - 0.5), 0, height - 1).astype(np.int64)
    iy1 = np.clip(np.floor(maxy - 0.5), -1, height - 1).astype(np.int64)
    bw, bh = ix1 - ix0 + 1, iy1 - iy0 + 1
    ok &= (bw > 0) & (bh > 0)
    ids = np.flatnonzero(ok)
    zbuf = np.full(height * width, np.inf, np.float64)
    tbuf = np.full(height * width, -1, np.int64)
    if ids.size:
        size = np.maximum(bw[ids], bh[ids])
        cls = np.ceil(np.log2(np.maximum(size, 1))).astype(np.int64)
        for c in np.unique(cls):
            grp = ids[cls == c]
            S = 1 << int(c)
            step = max(1, budget // (S * S))
            ar = np.arange(S)
            for s in range(0, grp.size, step):
                idx = grp[s:s + step]
                px = ix0[idx][:, None, None] + ar[None, None, :]
                py = iy0[idx][:, None, None] + ar[None, :, None]
                inbox = (px <= ix1[idx][:, None, None]) & (py <= iy1[idx][:, None, None])
                cx, cy = px + 0.5, py + 0.5
                a = area[idx][:, None, None]
                X0, Y0, X1, Y1, X2, Y2 = (t[idx][:, None, None] for t in (x0, y0, x1, y1, x2, y2))
                b0 = _edge(cx, cy, X1, Y1, X2, Y2) / a
                b1 = _edge(X0, Y0, cx, cy, X2, Y2) / a
                b2 = 1.0 - b0 - b1
                cov = inbox & (b0 >= -1e-9) & (b1 >= -1e-9) & (b2 >= -1e-9)
                if not cov.any():
                    continue
                n_i, r_i, c_i = np.nonzero(cov)
                t_id = idx[n_i]
                q = tw[t_id]
                depth = 1.0 / (b0[n_i, r_i, c_i] / q[:, 0] + b1[n_i, r_i, c_i] / q[:, 1] + b2[n_i, r_i, c_i] / q[:, 2])
                pix = py[n_i, r_i, 0] * width + px[n_i, 0, c_i]
                order = np.lexsort((t_id, depth, pix))
                pix_s, depth_s, tid_s = pix[order], depth[order], t_id[order]
                first = np.r_[True, pix_s[1:] != pix_s[:-1]]
                pix_s, depth_s, tid_s = pix_s[first], depth_s[first], tid_s[first]
                better = depth_s < zbuf[pix_s]
                zbuf[pix_s[better]] = depth_s[better]
                tbuf[pix_s[better]] = tid_s[better]
    hit = np.flatnonzero(tbuf >= 0)
    bary = np.zeros((height * width, 3), np.float32)
    depth = np.full(height * width, np.inf, np.float32)
    if hit.size:
        t_id = tbuf[hit]
        cx = (hit % width) + 0.5
        cy = (hit // width) + 0.5
        a = area[t_id]
        b0 = _edge(cx, cy, x1[t_id], y1[t_id], x2[t_id], y2[t_id]) / a
        b1 = _edge(x0[t_id], y0[t_id], cx, cy, x2[t_id], y2[t_id]) / a
        b2 = 1.0 - b0 - b1
        wv = tw[t_id]
        q0, q1, q2 = b0 / wv[:, 0], b1 / wv[:, 1], b2 / wv[:, 2]
        sm = q0 + q1 + q2
        bary[hit] = np.stack([q0 / sm, q1 / sm, q2 / sm], 1)
        depth[hit] = 1.0 / sm
    return tbuf.reshape(height, width), bary.reshape(height, width, 3), depth.reshape(height, width)


def render_camera(cam, verts: np.ndarray, faces: np.ndarray, height: int, width: Optional[int] = None):
    """Rasterise a mesh through a `Camera`.  Returns (mask (H,W) bool, depth (H,W) float32, tri (H,W) int64)."""
    width = width or height
    xy, w = project(cam, verts, height, width)
    tri, _bary, depth = rasterize(xy, w, faces, height, width)
    return tri >= 0, depth, tri


# ---------------------------------------------------------------------------------------------------- preview
def _rot(yaw: float, pitch: float) -> np.ndarray:
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rx @ ry
