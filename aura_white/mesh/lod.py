"""Level-of-detail chain and collision mesh generation.

LODs are produced by repeated quadric decimation of the SAME base mesh (never each other, so error doesn't compound)
down to a fraction of the original face count, then repaired so every LOD is independently watertight-checked.
Collision geometry is a convex hull by default (cheapest, always closed) with an optional voxel-remesh-based
"simplified" mode when a local Blender is available (better fit for concave shapes, at the cost of no longer being
convex - callers that need strict convexity for their physics engine should keep the default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .repair import RepairOptions, repair_mesh


@dataclass
class LODLevel:
    name: str                 # "LOD0", "LOD1", ...
    verts: np.ndarray
    faces: np.ndarray
    target_faces: int
    faces_actual: int
    watertight: bool


@dataclass
class CollisionMesh:
    verts: np.ndarray
    faces: np.ndarray
    kind: str                 # "convex_hull" | "voxel_simplified"
    volume: float
    faces_count: int


def _decimate(verts: np.ndarray, faces: np.ndarray, target: int):
    from ..texture import decimate as _dec

    v, f, _backend = _dec(verts, faces, max(4, int(target)))
    return np.asarray(v, np.float32), np.asarray(f, np.int64)


def generate_lods(verts: np.ndarray, faces: np.ndarray, levels: Sequence[float] = (1.0, 0.35, 0.12, 0.04),
                  repair: bool = True, min_faces: int = 12) -> List[LODLevel]:
    """`levels` are fractions of `len(faces)`; LOD0 is always the input mesh itself (fraction 1.0 is not
    re-decimated). Every level is decimated from the ORIGINAL mesh, not chained, so LOD3 is not a decimation of
    the already-lossy LOD2. Each level is independently topology-repaired when `repair=True`."""
    base_n = max(int(len(faces)), 1)
    out: List[LODLevel] = []
    for i, frac in enumerate(levels):
        target = max(min_faces, int(round(base_n * float(frac))))
        if i == 0 and frac >= 0.999:
            v, f = np.asarray(verts, np.float32), np.asarray(faces, np.int64)
        else:
            v, f = _decimate(verts, faces, target)
        watertight = True
        if repair:
            v, f, rep = repair_mesh(v, f, RepairOptions())
            watertight = bool(rep["watertight"])
        out.append(LODLevel(f"LOD{i}", v, f, target, int(len(f)), watertight))
    return out


def convex_hull_collision(verts: np.ndarray, faces: Optional[np.ndarray] = None) -> CollisionMesh:
    """A closed convex hull around the mesh - the cheapest, always-valid collision proxy; conservative (never
    smaller than the true shape) but can be much larger than it for concave/thin objects (a scythe's hull, for
    instance, fills in the space inside the blade's curve)."""
    from scipy.spatial import ConvexHull

    pts = np.asarray(verts, np.float64)
    hull = ConvexHull(pts)
    v = pts[hull.vertices]
    remap = -np.ones(len(pts), np.int64)
    remap[hull.vertices] = np.arange(len(hull.vertices))
    f = remap[hull.simplices]
    # ConvexHull's simplex winding is not guaranteed outward; fix it the same way repair.py would
    from .repair import orient_consistently

    f, _ = orient_consistently(v, f)
    return CollisionMesh(v.astype(np.float32), f.astype(np.int64), "convex_hull", float(hull.volume), int(len(f)))


def voxel_simplified_collision(verts: np.ndarray, faces: np.ndarray, voxel_size: float,
                               progress=None) -> Optional[CollisionMesh]:
    """A closer-fitting (non-convex) collision mesh via Blender's voxel remesher, when a local Blender is
    available; returns None (with a message sent to `progress`) otherwise - callers should fall back to
    `convex_hull_collision`."""
    from ..backends.blender_tools import voxel_remesh

    out = voxel_remesh(np.asarray(verts, np.float32), np.asarray(faces, np.int64), voxel_size, progress=progress)
    if out is None:
        return None
    v, f = out
    v, f, rep = repair_mesh(v, f, RepairOptions())
    vol = 0.0
    if rep["watertight"] and len(f):
        vv = v.astype(np.float64)
        vol = float(np.sum(np.einsum("ij,ij->i", vv[f[:, 0]], np.cross(vv[f[:, 1]], vv[f[:, 2]]))) / 6.0)
    return CollisionMesh(v, f, "voxel_simplified", abs(vol), int(len(f)))
