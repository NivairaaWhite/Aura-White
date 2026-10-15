"""Silhouette-exact relief meshing.

Given a soft mask and per-pixel half-thickness fields this builds a closed, watertight, double-sided surface whose
outline follows the mask's sub-pixel iso-contour (holes included) and whose front / back faces are height fields.

  * boundary vertices come from marching squares + Douglas-Peucker (dense where the outline curves, sparse where it is
    straight), interior vertices from a graded point cloud (dense near the outline, sparse in the middle)
  * triangulation = Delaunay (SciPy/Qhull) filtered against the mask, so nothing is invented outside the silhouette
  * front and back are joined by a wall along every boundary edge -> the mesh is closed even for thin plates

Coordinates are image space: x right, y DOWN (pixel-centre convention), z towards the viewer.  All lengths in pixels;
callers convert to world units.  NumPy + SciPy only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import morph


@dataclass
class ReliefMesh:
    verts: np.ndarray            # (M,3) float32   [x, y, z] px; first half front surface, second half back surface
    faces: np.ndarray            # (F,3) int64, outward-facing (counter-clockwise seen from outside)
    n_surface: int               # number of vertices per surface (front verts are 0..n-1, back verts n..2n-1)
    xy: np.ndarray               # (n,2) shared image coordinates of the surface vertices
    boundary: np.ndarray         # (n,) bool: vertex lies on the outline


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rod_profile(dt: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Half-thickness of a round rod (circular cross-section) of local radius `radius`."""
    u = np.clip(dt / np.maximum(radius, 1e-6), 0.0, 1.0)
    return radius * np.sqrt(u * (2.0 - u))


def plate_profile(dt: np.ndarray, half: float, bevel: float) -> np.ndarray:
    """Flat plate of half-thickness `half` with a smooth bevel of width `bevel` at the outline (knife edge)."""
    return half * smoothstep(dt / max(bevel, 1e-6))


def roof_profile(dt: np.ndarray, radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Faceted 'roof' (pyramid) profile: half-thickness grows linearly from the outline to the medial axis."""
    return height * np.clip(dt / np.maximum(radius, 1e-6), 0.0, 1.0)


def _sample(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    return ndi.map_coordinates(np.asarray(field, np.float32), [xy[:, 1], xy[:, 0]], order=1, mode="nearest")


def _points(alpha: np.ndarray, dt: np.ndarray, boundary_step: float, interior_step: Tuple[float, float],
            grade: float, tol: float) -> Tuple[np.ndarray, int]:
    loops = morph.contours(alpha, 0.5)
    bpts: List[np.ndarray] = []
    for lp in loops:
        if len(lp) < 4 or abs(morph.polygon_area(lp)) < 2.0:
            continue
        bpts.append(morph.simplify_closed(lp, tol, boundary_step * 2.0))
    if not bpts:
        return np.zeros((0, 2)), 0
    B = np.concatenate(bpts, 0)
    s0, smax = interior_step
    pts = [B]
    h, w = dt.shape
    k, s = 0, s0
    while s <= smax * 1.001:
        # this level serves where the desired spacing  s0 + grade*dt  lies in [s, 2s)
        d_lo = max((s - s0) / grade, 0.0) if k else 0.0
        d_hi = ((2 * s - s0) / grade) if s * 2 <= smax * 1.001 else 1e9
        off = 0.5 * s + 0.31 * s * (k % 2)
        gx, gy = np.meshgrid(np.arange(off, w, s), np.arange(off, h, s))
        gx, gy = gx.ravel(), gy.ravel()
        ix, iy = np.clip(gx.astype(int), 0, w - 1), np.clip(gy.astype(int), 0, h - 1)
        d = dt[iy, ix]
        keep = (d >= max(0.85 * s, 0.75 * boundary_step)) & (d >= d_lo) & (d < d_hi)
        if keep.any():
            pts.append(np.stack([gx[keep], gy[keep]], 1))
        k += 1
        s = s0 * (2 ** k)
    return np.concatenate(pts, 0), len(B)


def build_relief(alpha: np.ndarray, half_front: np.ndarray, half_back: Optional[np.ndarray] = None,
                 z_mid: Optional[np.ndarray] = None, boundary_step: float = 1.5,
                 interior_step: Tuple[float, float] = (2.5, 40.0), grade: float = 0.30,
                 simplify_tol: float = 0.22, edge_half: float = 0.15, max_faces: int = 0) -> Optional[ReliefMesh]:
    """Mesh the region alpha >= 0.5.  `half_front/back` are (H,W) half-thickness maps in px (>= 0); `z_mid` is the
    (H,W) mid-surface offset (default 0).  If `max_faces` is set the point density is relaxed until it fits."""
    from scipy.spatial import Delaunay

    a = np.asarray(alpha, np.float32)
    inside = a >= 0.5
    if not inside.any():
        return None
    dt = morph.edt(inside).astype(np.float32)
    hb_map = half_front if half_back is None else half_back
    step_scale = 1.0
    for _ in range(6):
        pts, nb = _points(a, dt, boundary_step * step_scale, (interior_step[0] * step_scale, interior_step[1]),
                          grade, simplify_tol * step_scale)
        if len(pts) < 3:
            return None
        rng = np.random.default_rng(7)
        jit = pts + rng.normal(0.0, 1e-3, pts.shape)
        tri = Delaunay(jit).simplices
        c = (jit[tri[:, 0]] + jit[tri[:, 1]] + jit[tri[:, 2]]) / 3.0
        ok = _sample(a, c) >= 0.5
        for i, j in ((0, 1), (1, 2), (2, 0)):
            ok &= _sample(a, 0.5 * (jit[tri[:, i]] + jit[tri[:, j]])) >= 0.3
        tri = tri[ok]
        if max_faces <= 0 or 2 * len(tri) + 2 * nb <= max_faces or step_scale > 6:
            break
        step_scale *= 1.35
    if len(tri) == 0:
        return None
    # orientation in the image plane (y down): make the shoelace area positive for every triangle
    p0, p1, p2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    tri = np.where((area < 0)[:, None], tri[:, [0, 2, 1]], tri)
    good = np.abs(area) > 1e-9
    tri = tri[good]

    # drop vertices that no triangle uses
    used = np.zeros(len(pts), bool)
    used[tri.ravel()] = True
    remap = -np.ones(len(pts), np.int64)
    remap[used] = np.arange(int(used.sum()))
    pts = pts[used]
    tri = remap[tri]
    n = len(pts)
    d_pts = _sample(dt, pts)

    hf = np.maximum(_sample(half_front, pts), 0.0)
    hb = np.maximum(_sample(hb_map, pts), 0.0)
    zm = _sample(z_mid, pts) if z_mid is not None else np.zeros(n, np.float32)
    on_edge = d_pts < 0.75
    hf = np.where(on_edge, np.maximum(hf, edge_half), hf)
    hb = np.where(on_edge, np.maximum(hb, edge_half), hb)
    front = np.concatenate([pts, (zm + hf)[:, None]], 1)
    back = np.concatenate([pts, (zm - hb)[:, None]], 1)
    verts = np.concatenate([front, back], 0).astype(np.float32)

    # wall along boundary edges (edges that belong to exactly one front triangle)
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], 0)
    key = np.minimum(e[:, 0], e[:, 1]) * n + np.maximum(e[:, 0], e[:, 1])
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bedges = e[cnt[inv] == 1]
    a_, b_ = bedges[:, 0], bedges[:, 1]
    faces = [tri, (tri[:, [0, 2, 1]] + n)]
    faces.append(np.stack([b_, a_, a_ + n], 1))
    faces.append(np.stack([a_ + n, b_ + n, b_], 1))
    faces = np.concatenate(faces, 0).astype(np.int64)
    # global orientation: the signed volume must be positive (outward normals)
    v = verts.astype(np.float64)
    vol = float(np.sum(np.einsum("ij,ij->i", v[faces[:, 0]], np.cross(v[faces[:, 1]], v[faces[:, 2]]))) / 6.0)
    if vol < 0:
        faces = faces[:, [0, 2, 1]]
    bnd = np.zeros(n, bool)
    bnd[bedges.ravel()] = True
    return ReliefMesh(verts, faces, n, pts.astype(np.float32), bnd)
