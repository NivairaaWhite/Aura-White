"""Geometry helpers for the native stack (numpy + scipy only): meshing a field, normalising a mesh, sampling its surface,
signed distances of real (possibly leaky) meshes, and the metrics used to compare two shapes."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

HALF = 0.9                                   # every shape lives in [-HALF, HALF]^3 (inside the [-1, 1] grid)


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def sdf_to_mesh(sdf: np.ndarray, level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Zero level set of an (R,R,R) signed-distance grid over [-1,1]^3 (negative = inside) -> (verts (V,3), faces (F,3))."""
    from ..mesh.surface_nets import surface_nets

    res = sdf.shape[0]
    pad = np.pad(-np.asarray(sdf, np.float32), 1, constant_values=-1.0)        # surface_nets: inside is > level
    v, f = surface_nets(pad, level=-level)
    v = (np.asarray(v, np.float64) - 1.0 + 0.5) / res * 2.0 - 1.0
    return v.astype(np.float64), np.asarray(f, np.int64)


def grid_coords(res: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    c = (np.arange(res) + 0.5) / res * (hi - lo) + lo
    return np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)


def normalize_mesh(v: np.ndarray, f: np.ndarray, half: float = HALF):
    """Centre the bounding box and scale its longest side to 2*half.  Returns (v', f, centre, scale)."""
    v = np.asarray(v, np.float64)
    lo, hi = v.min(0), v.max(0)
    c = (lo + hi) / 2.0
    s = 2.0 * half / max(float((hi - lo).max()), 1e-12)
    return (v - c) * s, np.asarray(f, np.int64), c, s


def surface_samples(v: np.ndarray, f: np.ndarray, n: int, rng: np.random.Generator):
    """n uniform surface points and their (unit) face normals."""
    tri = v[f]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    if area.sum() <= 0:
        raise ValueError("mesh has no area")
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    pts = a[:, None] * tri[idx, 0] + b[:, None] * tri[idx, 1] + c[:, None] * tri[idx, 2]
    nr = cr[idx]
    nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-12)
    return pts, nr


def mesh_sdf_grid(v: np.ndarray, f: np.ndarray, res: int = 96, seed: int = 0, n_samples: int = 300_000) -> np.ndarray:
    """Signed distance grid of a mesh inside [-1,1]^3.  Unsigned distance from dense surface samples (KD-tree); the sign
    from a flood fill of the exterior through voxels that are not part of the surface shell - works for open and
    self-intersecting meshes (a leaky shell only makes the inside smaller)."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pts, _ = surface_samples(np.asarray(v, np.float64), np.asarray(f, np.int64), n_samples, rng)
    P = grid_coords(res)
    dist, _ = cKDTree(pts).query(P, k=1, workers=-1)
    dist = dist.reshape(res, res, res)
    shell = dist < (2.0 / res) * 0.9
    lab, _n = ndi.label(~shell)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    exterior = np.isin(lab, border[border > 0])
    return (dist * np.where(exterior, 1.0, -1.0)).astype(np.float32)


def sample_grid(grid: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Trilinear lookup of an (R,R,R) grid over [-1,1]^3 at world points q (N,3); outside the cube -> edge value."""
    from scipy import ndimage as ndi

    res = grid.shape[0]
    idx = (np.asarray(q, np.float64) + 1.0) / 2.0 * res - 0.5
    return ndi.map_coordinates(grid, idx.T, order=1, mode="nearest").astype(np.float32)


# ------------------------------------------------------------------------------------------------ comparing shapes
def chamfer_fscore(pa: np.ndarray, pb: np.ndarray, taus=(0.02, 0.05)) -> Dict[str, float]:
    """Symmetric chamfer-L1 and F-score between two surface point sets (both in the same normalised frame)."""
    from scipy.spatial import cKDTree

    da, _ = cKDTree(pb).query(pa, k=1)
    db, _ = cKDTree(pa).query(pb, k=1)
    out = {"chamfer": float(0.5 * (da.mean() + db.mean()))}
    for t in taus:
        p, r = float((da < t).mean()), float((db < t).mean())
        out[f"f{t:g}"] = float(2 * p * r / max(p + r, 1e-12))
    return out


def mesh_chamfer(va, fa, vb, fb, n: int = 8000, seed: int = 0, taus=(0.02, 0.05)) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    pa, _ = surface_samples(np.asarray(va, np.float64), np.asarray(fa, np.int64), n, rng)
    pb, _ = surface_samples(np.asarray(vb, np.float64), np.asarray(fb, np.int64), n, rng)
    return chamfer_fscore(pa, pb, taus)


def volume_iou(sdf_a: np.ndarray, sdf_b: np.ndarray) -> float:
    a, b = np.asarray(sdf_a) < 0, np.asarray(sdf_b) < 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0
