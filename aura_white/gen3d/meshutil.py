"""Mesh <-> voxel utilities (numpy + scipy only).

`voxelize` fills a mesh by *winding number along rays* in all three axes and takes the majority, so overlapping shells,
a few holes and flipped components do not leave streaks.  `mesh_from_field` goes the other way (smoothed signed
distance -> surface nets), so a fused / carved volume becomes a mesh again without stair-steps.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

_JX, _JY = 0.013711, 0.007123            # sample jitter: keeps rays off exact shared edges (watertight counting)


def orient_outward(v: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Consistent, outward-facing winding for every shell (a generator may emit shells with mixed orientation)."""
    from ..mesh.repair import orient_consistently

    f2, _ = orient_consistently(np.asarray(v, np.float64), np.asarray(f, np.int64))
    return np.asarray(v, np.float64), np.asarray(f2, np.int64)


def _z_hits(P: np.ndarray, f: np.ndarray, n: int):
    """All ray/triangle hits for rays along +z through sample (i+.5+jx, j+.5+jy).  P is in grid-index units."""
    T = P[f]
    X, Y, Z = T[..., 0], T[..., 1], T[..., 2]
    area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (Y[:, 1] - Y[:, 0]) * (X[:, 2] - X[:, 0])
    ix0 = np.maximum(np.ceil(X.min(1) - 0.5 - _JX), 0).astype(np.int64)
    ix1 = np.minimum(np.floor(X.max(1) - 0.5 - _JX), n - 1).astype(np.int64)
    iy0 = np.maximum(np.ceil(Y.min(1) - 0.5 - _JY), 0).astype(np.int64)
    iy1 = np.minimum(np.floor(Y.max(1) - 0.5 - _JY), n - 1).astype(np.int64)
    nx, ny = ix1 - ix0 + 1, iy1 - iy0 + 1
    ok = (np.abs(area) > 1e-12) & (nx > 0) & (ny > 0)
    idx = np.flatnonzero(ok)
    pix_l, z_l, s_l = [], [], []
    if idx.size:
        span = np.maximum(nx[idx], ny[idx])
        cls = np.ceil(np.log2(np.maximum(span, 1))).astype(np.int64)
        for k in np.unique(cls):
            sel = idx[cls == k]
            s = 1 << int(k)
            step = max(1, 3_000_000 // (s * s))
            off = np.arange(s)
            for a in range(0, len(sel), step):
                c = sel[a:a + step]
                tx = ix0[c][:, None, None] + off[None, :, None]
                ty = iy0[c][:, None, None] + off[None, None, :]
                valid = (tx <= ix1[c][:, None, None]) & (ty <= iy1[c][:, None, None])
                px, py = tx + 0.5 + _JX, ty + 0.5 + _JY
                Ax, Ay, Bx, By, Cx, Cy = (X[c, 0][:, None, None], Y[c, 0][:, None, None], X[c, 1][:, None, None],
                                          Y[c, 1][:, None, None], X[c, 2][:, None, None], Y[c, 2][:, None, None])
                ar = area[c][:, None, None]
                wA = ((Bx - px) * (Cy - py) - (By - py) * (Cx - px)) / ar
                wB = ((px - Ax) * (Cy - Ay) - (py - Ay) * (Cx - Ax)) / ar
                wC = 1.0 - wA - wB
                inside = valid & (wA > 0) & (wB > 0) & (wC > 0)
                ci, ai, bi = np.nonzero(inside)
                if ci.size == 0:
                    continue
                cc = c[ci]
                z = wA[inside] * Z[cc, 0] + wB[inside] * Z[cc, 1] + wC[inside] * Z[cc, 2]
                pix_l.append((ix0[cc] + ai) * n + (iy0[cc] + bi))
                z_l.append(z)
                s_l.append(np.where(area[cc] < 0, 1, -1))          # normal facing the ray origin = entering
    if not pix_l:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
    return np.concatenate(pix_l), np.concatenate(z_l), np.concatenate(s_l)


def _occupancy_along_z(P: np.ndarray, f: np.ndarray, n: int) -> np.ndarray:
    pix, z, sg = _z_hits(P, f, n)
    occ = np.zeros((n * n, n), bool)
    if pix.size == 0:
        return occ.reshape(n, n, n)
    o = np.lexsort((z, pix))
    pix, z, sg = pix[o], z[o], sg[o]
    start = np.r_[True, pix[1:] != pix[:-1]]
    gid = np.cumsum(start) - 1
    cs = np.cumsum(sg)
    base = np.r_[0, cs[np.flatnonzero(start)[1:] - 1]]
    w = cs - base[gid]
    nxt = np.r_[pix[1:] == pix[:-1], False]
    use = np.flatnonzero((w > 0) & nxt)
    if use.size:
        k0 = np.clip(np.ceil(z[use] - 0.5), 0, n).astype(np.int64)
        k1 = np.clip(np.ceil(z[use + 1] - 0.5), 0, n).astype(np.int64)
        D = (np.bincount(pix[use] * (n + 1) + k0, minlength=n * n * (n + 1)).astype(np.int32)
             - np.bincount(pix[use] * (n + 1) + k1, minlength=n * n * (n + 1)).astype(np.int32)).reshape(n * n, n + 1)
        occ = np.cumsum(D, axis=1)[:, :n] > 0
    return occ.reshape(n, n, n)


def voxelize(v: np.ndarray, f: np.ndarray, n: int, lo: np.ndarray, hi: float) -> np.ndarray:
    """Occupancy (n,n,n) indexed [ix,iy,iz] of a (closed, outward-facing) mesh.  The grid is the cube starting at the
    corner `lo` with side `hi`; voxel k's centre is lo + (k + .5) * hi / n.  Majority of the three axis sweeps."""
    h = hi / n
    P = (np.asarray(v, np.float64) - np.asarray(lo, np.float64)) / h
    f = np.asarray(f, np.int64)
    oz = _occupancy_along_z(P, f, n)
    ox = _occupancy_along_z(P[:, [1, 2, 0]], f, n).transpose(2, 0, 1)
    oy = _occupancy_along_z(P[:, [2, 0, 1]], f, n).transpose(1, 2, 0)
    return (oz.astype(np.uint8) + ox + oy) >= 2


def smooth_field(occ: np.ndarray, sigma: float = 0.9) -> np.ndarray:
    """Occupancy -> smooth field with inside > 0 (a signed distance in voxels, lightly blurred)."""
    from scipy import ndimage as ndi

    pad = np.pad(occ, 2, constant_values=False)
    d_out = ndi.distance_transform_edt(~pad).astype(np.float32)
    d_in = ndi.distance_transform_edt(pad).astype(np.float32)
    field = d_in - d_out
    if sigma > 0:
        field = ndi.gaussian_filter(field, sigma)
    return field


def mesh_from_field(field: np.ndarray, lo: np.ndarray, hi: float, pad: int = 2):
    """Surface-nets mesh (world coordinates) of `field == 0` for a field made by `smooth_field`."""
    from ..mesh.surface_nets import surface_nets

    n = field.shape[0] - 2 * pad
    h = hi / n
    vi, fi = surface_nets(field, 0.0)
    if len(vi) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    world = np.asarray(lo, np.float64) + (np.asarray(vi, np.float64) - pad + 0.5) * h
    return world, np.asarray(fi, np.int64)


def grid_for(v: np.ndarray, margin: float = 0.06) -> Tuple[np.ndarray, float]:
    """(corner, side) of a cube that holds the mesh with a margin."""
    lo, hi = v.min(0), v.max(0)
    c = (lo + hi) / 2.0
    side = float((hi - lo).max()) * (1.0 + 2.0 * margin)
    return c - side / 2.0, side


def component_labels(f: np.ndarray, n_vertices: int) -> np.ndarray:
    """Connected-component id per face."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(f, np.int64)
    a = np.r_[f[:, 0], f[:, 1], f[:, 2]]
    b = np.r_[f[:, 1], f[:, 2], f[:, 0]]
    g = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n_vertices, n_vertices))
    _, lab = connected_components(g, directed=False)
    return lab[f[:, 0]]
