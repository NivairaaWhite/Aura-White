"""High-poly -> low-poly detail bake (NumPy + SciPy only): a tangent-space normal map and a cavity term.

The supplied mesh usually has far more triangles than a game wants.  The game mesh (LOD0) is a decimated copy; the detail
the decimation removed (folds, bricks, rock facets, cloth) is carried over as a normal map, exactly as in a classic
high-to-low bake: every texel looks up the nearest point of the full-resolution surface and stores its normal in the
tangent frame of the game mesh.  glTF conventions: red = +u, green = image-up, blue = along the surface normal.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .camera import rasterize_ortho


def _vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def bake_hi_maps(hi_v: np.ndarray, hi_f: np.ndarray, lo_v: np.ndarray, lo_f: np.ndarray, lo_uv: np.ndarray,
                 lo_n: np.ndarray, lo_t: np.ndarray, size: Tuple[int, int], max_dist: Optional[float] = None,
                 samples_per_face: int = 2, strength: float = 1.0, min_align: float = 0.25, seed: int = 0):
    """-> (normal map uint8 (h,w,3), cavity float32 (h,w) in 0..1 with 1 = open surface).
    `lo_uv` is in 0..1 atlas coordinates; `size` = (width, height) of the maps; `lo_t` are per-vertex tangents (V,4)."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    w, h = int(size[0]), int(size[1])
    hv = np.asarray(hi_v, np.float64)
    hf = np.asarray(hi_f, np.int64)
    diag = float(np.linalg.norm(hv.max(0) - hv.min(0)))
    cap = float(max_dist) if max_dist is not None else 0.03 * diag
    hn = _vertex_normals(hv, hf)
    rng = np.random.default_rng(seed)
    b = rng.dirichlet((1.0, 1.0, 1.0), size=(len(hf), samples_per_face))
    pts = (hv[hf][:, None] * b[..., None]).sum(2).reshape(-1, 3)
    nrm = (hn[hf][:, None] * b[..., None]).sum(2).reshape(-1, 3)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    tree = cKDTree(pts)

    xy = np.asarray(lo_uv, np.float64) * np.array([w, h], np.float64)
    tid, bary, _ = rasterize_ortho(xy, np.zeros(len(xy)), np.asarray(lo_f, np.int64), h, w)
    hit = tid >= 0
    normal = np.zeros((h, w, 3), np.float32)
    cav = np.ones((h, w), np.float32)
    if not hit.any():
        return np.full((h, w, 3), (128, 128, 255), np.uint8), cav
    hh, ww = np.nonzero(hit)
    lv = np.asarray(lo_v, np.float64)
    lf = np.asarray(lo_f, np.int64)
    ln = np.asarray(lo_n, np.float64)
    lt = np.asarray(lo_t, np.float64)
    CH = 300_000
    for s0 in range(0, len(hh), CH):
        sl = slice(s0, s0 + CH)
        t = tid[hh[sl], ww[sl]]
        bb = bary[hh[sl], ww[sl]].astype(np.float64)
        tri = lf[t]
        P = (lv[tri] * bb[..., None]).sum(1)
        N = (ln[tri] * bb[..., None]).sum(1)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        T = (lt[tri][..., :3] * bb[..., None]).sum(1)
        T = T - N * (N * T).sum(1, keepdims=True)
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
        sgn = np.where(lt[tri[:, 0], 3] < 0, -1.0, 1.0)[:, None]
        B = np.cross(N, T) * sgn
        d, j = tree.query(P, workers=-1)
        n_hi = nrm[j]
        ok = (d <= cap) & ((n_hi * N).sum(1) >= min_align)
        n_hi = np.where(ok[:, None], n_hi, N)
        nt = np.stack([(n_hi * T).sum(1) * strength, (n_hi * B).sum(1) * strength, (n_hi * N).sum(1)], 1)
        nt /= np.maximum(np.linalg.norm(nt, axis=1, keepdims=True), 1e-12)
        normal[hh[sl], ww[sl]] = nt
        disp = ((pts[j] - P) * N).sum(1)                                  # < 0: the full surface lies below the game mesh
        cav[hh[sl], ww[sl]] = np.where(ok, 1.0 - np.clip(-disp / (0.004 * diag), 0.0, 1.0), 1.0)
    if not hit.all():
        _, (iy, ix) = ndi.distance_transform_edt(~hit, return_indices=True)
        normal = normal[iy, ix]
        cav = cav[iy, ix]
    img = np.clip((normal * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img, cav
