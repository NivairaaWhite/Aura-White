"""Naive Surface Nets isosurface extraction in vectorised numpy.

Replaces `torchmcubes` (a compiled, unmaintained git dependency - the thing that broke on new
Python versions) with ~80 lines that run anywhere numpy runs. Compared with marching cubes it
gives one vertex per active cell (about half the vertices), better-shaped triangles and
smoother surfaces, and the mesh is closed and consistently oriented by construction.
"""
from __future__ import annotations

import numpy as np


def resolve_ambiguous_faces(vol: np.ndarray, level: float, max_passes: int = 12) -> np.ndarray:
    """Make the mesh edge-manifold.

    A grid face whose four corners alternate inside/outside ('checkerboard') makes surface nets
    put four triangles on one edge. Flip the corner closest to the level on each such face
    (a sub-voxel change) until none are left. Works on and returns a copy of `vol`."""
    vol = vol.copy()
    lvl = np.float32(level)
    up, down = np.nextafter(lvl, np.float32(np.inf)), np.nextafter(lvl, np.float32(-np.inf))
    dims = vol.shape
    touched = np.zeros(dims, dtype=bool)  # corners already flipped once are not flipped back
    for _ in range(max_passes):
        inside = vol > level
        fixed_any = False
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3

            def sl(db, dc):
                s = [slice(None)] * 3
                s[b], s[c] = slice(db, dims[b] - 1 + db), slice(dc, dims[c] - 1 + dc)
                return tuple(s)

            s00, s01, s10, s11 = inside[sl(0, 0)], inside[sl(0, 1)], inside[sl(1, 0)], inside[sl(1, 1)]
            amb = (s00 == s11) & (s01 == s10) & (s00 != s01)
            idx = np.argwhere(amb)
            if idx.shape[0] == 0:
                continue
            fixed_any = True
            corners = []
            for db, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                ci = idx.copy()
                ci[:, b] += db
                ci[:, c] += dc
                corners.append(ci)
            vals = np.stack([np.where(touched[tuple(ci.T)], np.inf, np.abs(vol[tuple(ci.T)] - lvl)) for ci in corners], axis=1)
            pick = np.argmin(vals, axis=1)
            for k in range(4):
                sel = pick == k
                if not sel.any():
                    continue
                ci = corners[k][sel]
                cur = vol[tuple(ci.T)]
                vol[tuple(ci.T)] = np.where(cur > lvl, down, up)
                touched[tuple(ci.T)] = True
        if not fixed_any:
            break
    return vol


def surface_nets(volume: np.ndarray, level: float = 0.0, manifold: bool = True):
    """Extract the iso-surface `volume == level` where `volume > level` is *inside*.

    volume : (X, Y, Z) array. Returns (vertices float32 (V,3) in index coordinates,
             faces int32 (F,3)) with counter-clockwise winding seen from outside.
    manifold: repair checkerboard faces first so every edge has exactly two triangles.
    """
    vol = np.ascontiguousarray(volume, dtype=np.float32)
    if vol.ndim != 3 or min(vol.shape) < 2:
        raise ValueError("volume must be a 3-D array with every side >= 2")
    if manifold:
        vol = resolve_ambiguous_faces(vol, level)
    dims = vol.shape
    inside = vol > level
    cdims = (dims[0] - 1, dims[1] - 1, dims[2] - 1)  # cells per axis
    cstride = (cdims[1] * cdims[2], cdims[2], 1)

    acc_ids, acc_pos, quads, flips = [], [], [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3  # right-handed: a = b x c, so CCW in (b,c) faces +a
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a], hi[a] = slice(0, dims[a] - 1), slice(1, dims[a])
        lo, hi = tuple(lo), tuple(hi)
        i0, i1 = inside[lo], inside[hi]
        crossing = i0 != i1
        idx = np.argwhere(crossing)  # (n,3) edge coordinates (index along `a` == owning cell)
        if idx.shape[0] == 0:
            continue
        v0, v1 = vol[lo][crossing], vol[hi][crossing]
        t = np.clip((level - v0) / np.where(v1 == v0, 1.0, v1 - v0), 0.0, 1.0)
        pos = idx.astype(np.float32)
        pos[:, a] += t.astype(np.float32)
        outward_plus = i0[crossing]  # inside on the low side -> outward normal is +a

        ring = []
        all_ok = np.ones(idx.shape[0], dtype=bool)
        for db, dc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):  # CCW in the (b,c) plane
            cb, cc = idx[:, b] + db, idx[:, c] + dc
            ok = (cb >= 0) & (cb < cdims[b]) & (cc >= 0) & (cc < cdims[c])
            flat = idx[:, a] * cstride[a] + cb * cstride[b] + cc * cstride[c]
            all_ok &= ok
            acc_ids.append(flat[ok])
            acc_pos.append(pos[ok])
            ring.append(flat)
        quads.append(np.stack(ring, axis=1)[all_ok])
        flips.append(~outward_plus[all_ok])

    if not acc_ids:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    ids = np.concatenate(acc_ids)
    pos = np.concatenate(acc_pos)
    uniq, inv = np.unique(ids.ravel(), return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    verts = np.stack([np.bincount(inv, weights=pos[:, d], minlength=uniq.size) / cnt for d in range(3)], axis=1)
    verts = verts.astype(np.float32)

    quad = np.concatenate(quads) if quads else np.zeros((0, 4), np.int64)
    if quad.shape[0] == 0:
        return verts, np.zeros((0, 3), np.int32)
    flip = np.concatenate(flips)
    q = np.searchsorted(uniq, quad)  # cell id -> vertex index
    q[flip] = q[flip][:, [0, 3, 2, 1]]  # reverse winding where the outward side is -axis

    # split each quad along its shorter diagonal (better triangles, no long slivers)
    d02 = np.sum((verts[q[:, 0]] - verts[q[:, 2]]) ** 2, axis=1)
    d13 = np.sum((verts[q[:, 1]] - verts[q[:, 3]]) ** 2, axis=1)
    use02 = d02 <= d13
    tri_a = np.where(use02[:, None], q[:, [0, 1, 2]], q[:, [1, 2, 3]])
    tri_b = np.where(use02[:, None], q[:, [0, 2, 3]], q[:, [1, 3, 0]])
    faces = np.concatenate([tri_a, tri_b], axis=0).astype(np.int32)
    return verts, faces
