from __future__ import annotations

import numpy as np


def remove_unreferenced(verts: np.ndarray, faces: np.ndarray, *extra):
    """Drop vertices no face uses and re-index. `extra` per-vertex arrays are filtered alongside."""
    if faces.shape[0] == 0:
        return (verts[:0], faces, *[e[:0] for e in extra])
    used = np.zeros(verts.shape[0], dtype=bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    return (verts[used], remap[faces].astype(np.int32), *[e[used] for e in extra])


def connected_components(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """Component label per vertex (vertices connected through shared faces)."""
    if faces.shape[0] == 0:
        return np.arange(n_verts)
    try:  # fast path when scipy happens to be installed - never required
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components as _cc

        f = faces
        r = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        c = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
        g = coo_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(n_verts, n_verts))
        return _cc(g, directed=False)[1]
    except Exception:
        pass
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    label = np.arange(n_verts)
    for _ in range(100000):
        m = np.minimum(np.minimum(label[a], label[b]), label[c])
        new = label.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        np.minimum.at(new, c, m)
        new = new[new]  # pointer jumping: converges in ~log(diameter) rounds
        if np.array_equal(new, label):
            break
        label = new
    return label


def keep_large_components(verts, faces, colors=None, min_ratio: float = 0.02):
    """Remove floating fragments: keep components with >= min_ratio x (faces of the largest)."""
    if faces.shape[0] == 0 or min_ratio <= 0:
        return verts, faces, colors
    lab = connected_components(faces, verts.shape[0])
    _, comp = np.unique(lab, return_inverse=True)
    comp = comp.ravel()
    fcount = np.bincount(comp[faces[:, 0]])
    keep_comp = fcount >= max(1.0, min_ratio * fcount.max())
    keep_face = keep_comp[comp[faces[:, 0]]]
    if keep_face.all():
        return verts, faces, colors
    faces = faces[keep_face]
    if colors is None:
        v, f = remove_unreferenced(verts, faces)
        return v, f, None
    v, f, c = remove_unreferenced(verts, faces, colors)
    return v, f, c


def _edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    return np.unique(e, axis=0)


def taubin_smooth(verts: np.ndarray, faces: np.ndarray, iters: int = 2, lam: float = 0.5, mu: float = -0.53):
    """Volume-preserving smoothing (Taubin lambda|mu): removes voxel stair-stepping without shrinking."""
    if iters <= 0 or faces.shape[0] == 0:
        return verts
    n = verts.shape[0]
    e = _edges(faces)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.maximum(np.bincount(src, minlength=n), 1).astype(np.float64)
    v = verts.astype(np.float64)

    def step(v, w):
        mean = np.stack([np.bincount(src, weights=v[dst, d], minlength=n) for d in range(3)], axis=1) / deg[:, None]
        return v + w * (mean - v)

    for _ in range(iters):
        v = step(v, lam)
        v = step(v, mu)
    return v.astype(np.float32)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    n = verts.shape[0]
    if faces.shape[0] == 0:
        return np.zeros((n, 3), np.float32)
    p = verts.astype(np.float64)
    fn = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])  # length = 2*area
    out = np.zeros((n, 3))
    for k in range(3):
        for d in range(3):
            out[:, d] += np.bincount(faces[:, k], weights=fn[:, d], minlength=n)
    ln = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(ln, 1e-12)).astype(np.float32)


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    p = verts.astype(np.float64)
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
