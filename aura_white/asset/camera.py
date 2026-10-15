"""Orthographic reference camera and an exact NumPy z-buffer for it.

World frame is glTF's: +y up, +z towards the viewer, +x to the right.  Lengths are in *source-image pixels*; the
exporter scales to metres.  The reference camera looks at the object from azimuth 0 at `elev` degrees above the horizon
(0 = a straight front view, 30 = the usual 'isometric game art' angle).

    screen right  = +x
    screen up     = u = (0, cos e, -sin e)
    to the camera = c = (0, sin e,  cos e)          ("t" below is the coordinate along c, larger = closer)

A surface seen by the reference camera is therefore a depth field t(x_px, y_px): that is what lets every layer keep
the reference view pixel-exact - vertices are slid along view rays (changing t) and never leave their pixel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class OrthoCamera:
    width: int
    height: int
    elev: float = 0.0                       # degrees above the horizon
    cx: float = 0.0                         # world x that sits at the image centre column
    cy: float = 0.0                         # screen-up coordinate that sits at the image centre row

    @property
    def e(self) -> float:
        return np.radians(self.elev)

    @property
    def to_camera(self) -> np.ndarray:
        e = self.e
        return np.array([0.0, np.sin(e), np.cos(e)])

    @property
    def screen_up(self) -> np.ndarray:
        e = self.e
        return np.array([0.0, np.cos(e), -np.sin(e)])

    def project(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World (N,3) -> (x_px, y_px, t)."""
        P = np.asarray(P, np.float64)
        e = self.e
        sy = P[:, 1] * np.cos(e) - P[:, 2] * np.sin(e)
        t = P[:, 1] * np.sin(e) + P[:, 2] * np.cos(e)
        return P[:, 0] - self.cx + self.width / 2.0, self.height / 2.0 + self.cy - sy, t

    def unproject(self, x_px, y_px, t) -> np.ndarray:
        e = self.e
        X = np.asarray(x_px, np.float64) - self.width / 2.0 + self.cx
        sy = self.height / 2.0 + self.cy - np.asarray(y_px, np.float64)
        t = np.asarray(t, np.float64)
        return np.stack([X, sy * np.cos(e) + t * np.sin(e), -sy * np.sin(e) + t * np.cos(e)], -1)

    def ground_points(self, x_px, y_px, Y=0.0) -> np.ndarray:
        """World points on the horizontal plane `Y` that project to pixels (x_px, y_px) -> (N,3)."""
        e = self.e
        x_px, y_px = np.atleast_1d(np.asarray(x_px, np.float64)), np.atleast_1d(np.asarray(y_px, np.float64))
        sy = self.height / 2.0 + self.cy - y_px
        t = (np.asarray(Y, np.float64) - sy * np.cos(e)) / max(np.sin(e), 1e-9)
        return self.unproject(x_px, y_px, t)

    def ground_point(self, x_px: float, y_px: float, Y: float = 0.0) -> np.ndarray:
        return self.ground_points(x_px, y_px, Y)[0]

    def ramp(self, plane: str) -> np.ndarray:
        """Per-row slope dt/d(screen-up) of a base plane seen by this camera: 'facing' (0), 'upright' (a vertical wall,
        +tan e) or 'ground' (a horizontal floor, -cot e)."""
        e = self.e
        if plane == "facing":
            return np.float64(0.0)
        if plane == "upright":
            return np.float64(np.tan(e))
        if plane == "ground":
            return np.float64(-1.0 / max(np.tan(e), 1e-6))
        raise ValueError(plane)


# ------------------------------------------------------------------------------------------------ raster
def _edge(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def rasterize_ortho(xy: np.ndarray, depth: np.ndarray, tris: np.ndarray, height: int, width: int,
                    budget: int = 1_500_000, nearest_is_larger: bool = False):
    """Exact affine z-buffer (no perspective division).  `depth` is any per-vertex scalar; the smaller value wins
    unless `nearest_is_larger`.  Returns (tri (H,W) int64, -1 = empty; bary (H,W,3) float32; depth (H,W) float32,
    NaN = empty).  Pixel centres sit at +0.5, y down - the convention of the rest of the package."""
    xy = np.asarray(xy, np.float64)
    d = np.asarray(depth, np.float64)
    if nearest_is_larger:
        d = -d
    tris = np.asarray(tris, np.int64)
    p = xy[tris]
    td = d[tris]
    x0, y0, x1, y1, x2, y2 = p[:, 0, 0], p[:, 0, 1], p[:, 1, 0], p[:, 1, 1], p[:, 2, 0], p[:, 2, 1]
    area = _edge(x0, y0, x1, y1, x2, y2)
    ok = np.isfinite(p).all((1, 2)) & np.isfinite(td).all(1) & (np.abs(area) > 1e-12)
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
                q = td[t_id]
                z = b0[n_i, r_i, c_i] * q[:, 0] + b1[n_i, r_i, c_i] * q[:, 1] + b2[n_i, r_i, c_i] * q[:, 2]
                pix = py[n_i, r_i, 0] * width + px[n_i, 0, c_i]
                order = np.lexsort((t_id, z, pix))
                pix_s, z_s, tid_s = pix[order], z[order], t_id[order]
                first = np.r_[True, pix_s[1:] != pix_s[:-1]]
                pix_s, z_s, tid_s = pix_s[first], z_s[first], tid_s[first]
                better = z_s < zbuf[pix_s]
                zbuf[pix_s[better]] = z_s[better]
                tbuf[pix_s[better]] = tid_s[better]
    hit = np.flatnonzero(tbuf >= 0)
    bary = np.zeros((height * width, 3), np.float32)
    depth_out = np.full(height * width, np.nan, np.float32)
    if hit.size:
        t_id = tbuf[hit]
        cx = (hit % width) + 0.5
        cy = (hit // width) + 0.5
        a = area[t_id]
        b0 = _edge(cx, cy, x1[t_id], y1[t_id], x2[t_id], y2[t_id]) / a
        b1 = _edge(x0[t_id], y0[t_id], cx, cy, x2[t_id], y2[t_id]) / a
        b2 = 1.0 - b0 - b1
        bary[hit] = np.stack([b0, b1, b2], 1)
        z = b0 * td[t_id, 0] + b1 * td[t_id, 1] + b2 * td[t_id, 2]
        depth_out[hit] = (-z if nearest_is_larger else z)
    return tbuf.reshape(height, width), bary.reshape(height, width, 3), depth_out.reshape(height, width)


def render_view(cam: OrthoCamera, verts: np.ndarray, faces: np.ndarray, height: int = None, width: int = None):
    """Depth-test a world-space mesh from the reference camera.  Returns (tri, bary, t) with t = coordinate towards the
    camera (larger = closer; NaN = background)."""
    height = height or cam.height
    width = width or cam.width
    sx, sy, t = cam.project(verts)
    k = width / cam.width
    return rasterize_ortho(np.stack([sx * k, sy * k], 1), t, faces, height, width, nearest_is_larger=True)
