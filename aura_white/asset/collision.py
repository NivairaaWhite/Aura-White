"""Collision proxies: sets of convex hulls (what every physics engine wants), kept small."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def reduce_points(pts: np.ndarray, max_points: int = 96) -> np.ndarray:
    """Farthest-point sampling of the hull vertices so every hull stays tiny (a few hundred triangles at most)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) <= max_points:
        return pts
    sel = [int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1)))]
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for _ in range(max_points - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return pts[sel]


def hull(points: np.ndarray, max_points: int = 64):
    """-> (verts (n,3) float32, faces (m,3) int64, volume) of the convex hull, outward-wound."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(points, np.float64)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 4:
        raise ValueError("a hull needs at least four points")
    h0 = ConvexHull(pts)
    pts = reduce_points(pts[h0.vertices], max_points)
    h = ConvexHull(pts)
    v = pts[h.vertices]
    remap = -np.ones(len(pts), np.int64)
    remap[h.vertices] = np.arange(len(h.vertices))
    f = remap[h.simplices]
    c = v.mean(0)                                                    # orient every triangle away from the centroid
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = np.einsum("ij,ij->i", n, v[f[:, 0]] - c) < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    return v.astype(np.float32), f.astype(np.int64), float(h.volume)


def kmeans_hulls(points: np.ndarray, k: int, seed: int = 0, iters: int = 12) -> List[np.ndarray]:
    """Split the points into k spatial clusters (k-means) -> list of point sets (one hull each)."""
    pts = np.asarray(points, np.float64)
    k = int(max(1, min(k, len(pts) // 8)))
    if k == 1:
        return [pts]
    rng = np.random.default_rng(seed)
    cent = pts[rng.choice(len(pts), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((pts[:, None, :] - cent[None]) ** 2).sum(-1), axis=1) if len(pts) * k < 4e6 else \
            np.argmin(np.stack([((pts - c) ** 2).sum(1) for c in cent], 1), axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                cent[j] = pts[m].mean(0)
    return [pts[lab == j] for j in range(k) if (lab == j).sum() >= 4]


def sector_points(points: np.ndarray, centre_xz: Tuple[float, float], n: int) -> List[np.ndarray]:
    """Angular sectors around a vertical axis (wedges of a ring)."""
    pts = np.asarray(points, np.float64)
    th = np.arctan2(pts[:, 0] - centre_xz[0], pts[:, 2] - centre_xz[1])
    idx = np.floor((th + np.pi) / (2 * np.pi) * n).astype(int) % n
    out = []
    for j in range(n):
        # overlap the neighbours by one point ring so the wedges touch
        m = idx == j
        if m.sum() >= 4:
            out.append(pts[m])
    return out


def build_hulls(groups: Sequence[np.ndarray], max_points: int = 64):
    res = []
    for g in groups:
        try:
            res.append(hull(g, max_points))
        except Exception:                                             # coplanar / degenerate group
            continue
    return res
