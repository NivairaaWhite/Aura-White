"""Signed-distance volumes: procedural shapes, mesh voxelisation, single-view observation masks, meshing.

Grid convention: R^3 cells over [-1, 1]^3, cell centre i -> (i + 0.5) / R * 2 - 1.  Axis 0 is x; the viewer sits at +x
(the canonical Aura White convention: the artwork is seen from +x)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

TRUNC = 3.0          # truncation of the SDF, in voxel widths


def grid_points(res: int) -> np.ndarray:
    c = (np.arange(res) + 0.5) / res * 2.0 - 1.0
    return np.stack(np.meshgrid(c, c, c, indexing="ij"), -1)                # (R,R,R,3)


def to_tsdf(sdf: np.ndarray, res: int) -> np.ndarray:
    """Truncated, normalised SDF in [-1, 1] (negative = inside)."""
    return np.clip(sdf / (TRUNC * 2.0 / res), -1.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------------------ procedural
def _rot(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q) + 1e-9
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _primitive(kind: str, p: np.ndarray, size: np.ndarray) -> np.ndarray:
    if kind == "sphere":
        return np.linalg.norm(p, axis=-1) - size[0]
    if kind == "box":
        q = np.abs(p) - size[None, None, None, :]
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
    if kind == "cylinder":
        d = np.stack([np.linalg.norm(p[..., :2], axis=-1) - size[0], np.abs(p[..., 2]) - size[1]], -1)
        return np.minimum(d.max(-1), 0) + np.linalg.norm(np.maximum(d, 0), axis=-1)
    if kind == "capsule":
        z = np.clip(p[..., 2], -size[1], size[1])
        return np.linalg.norm(np.stack([p[..., 0], p[..., 1], p[..., 2] - z], -1), axis=-1) - size[0]
    if kind == "torus":
        q = np.stack([np.linalg.norm(p[..., :2], axis=-1) - size[0], p[..., 2]], -1)
        return np.linalg.norm(q, axis=-1) - size[1]
    raise ValueError(kind)


def random_shape(res: int, rng: np.random.Generator, n_parts: Optional[int] = None) -> np.ndarray:
    """A random union of 1-5 primitives (uniformly scaled, rotated, translated) -> exact-ish SDF (R,R,R)."""
    P = grid_points(res)
    n = n_parts or int(rng.integers(1, 6))
    kinds = ["sphere", "box", "cylinder", "capsule", "torus"]
    sdf = None
    for _ in range(n):
        k = kinds[int(rng.integers(len(kinds)))]
        s = float(rng.uniform(0.15, 0.5))
        size = {"sphere": np.array([s]), "box": rng.uniform(0.6, 1.0, 3) * s * 1.1,
                "cylinder": np.array([s * rng.uniform(0.5, 0.9), s * rng.uniform(0.6, 1.4)]),
                "capsule": np.array([s * rng.uniform(0.3, 0.6), s * rng.uniform(0.5, 1.3)]),
                "torus": np.array([s * 0.8, s * rng.uniform(0.2, 0.4)])}[k]
        R = _rot(rng)
        t = rng.uniform(-0.35, 0.35, 3) * (1.0 if n > 1 else 0.3)
        d = _primitive(k, (P - t) @ R, size)
        sdf = d if sdf is None else np.minimum(sdf, d)
    return sdf.astype(np.float32)


# ------------------------------------------------------------------------------------------------ meshes
def _surface_samples(v: np.ndarray, f: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    return a[:, None] * tri[idx, 0] + b[:, None] * tri[idx, 1] + c[:, None] * tri[idx, 2]


def mesh_to_sdf(verts: np.ndarray, faces: np.ndarray, res: int, seed: int = 0, n_samples: int = 400_000) -> np.ndarray:
    """Signed distance of a mesh inside [-1,1]^3 (the mesh must already fit).  Unsigned distance from surface samples
    (KD-tree); the sign from a flood fill of the exterior through voxels that are not part of the surface shell, so
    it works for open and self-intersecting meshes too (a leaky shell just makes the inside smaller)."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pts = _surface_samples(np.asarray(verts, np.float64), np.asarray(faces, np.int64), n_samples, rng)
    P = grid_points(res).reshape(-1, 3)
    dist, _ = cKDTree(pts).query(P, k=1, workers=-1)
    dist = dist.reshape(res, res, res)
    shell = dist < (1.0 / res) * 0.9
    empty = ~shell
    lab, _n = ndi.label(empty)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    exterior = np.isin(lab, border[border > 0])
    sign = np.where(exterior, 1.0, -1.0)
    return (dist * sign).astype(np.float32)


def observed_mask(sdf: np.ndarray, thickness: int = 2, axis: int = 0, from_positive: bool = True) -> np.ndarray:
    """Voxels a depth camera at +axis would observe: everything in front of the first surface it hits plus a
    `thickness`-voxel shell behind it.  Everything else (the hidden volume) is what the model must generate."""
    inside = sdf < 0
    x = np.moveaxis(inside, axis, 0)
    if from_positive:
        x = x[::-1]
    cum = np.cumsum(x, axis=0)
    obs = cum <= thickness
    if from_positive:
        obs = obs[::-1]
    return np.moveaxis(obs, 0, axis)


def sdf_to_mesh(sdf: np.ndarray, level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Mesh of the zero level set, in [-1,1]^3 coordinates (uses Aura White's own surface-nets mesher)."""
    from ..mesh.surface_nets import surface_nets

    res = sdf.shape[0]
    pad = np.pad(-sdf.astype(np.float32), 1, constant_values=-1.0)          # surface_nets: inside is > level
    v, f = surface_nets(pad, level=-level)
    v = (np.asarray(v, np.float64) - 1.0 + 0.5) / res * 2.0 - 1.0
    return v.astype(np.float32), np.asarray(f, np.int64)
