"""Checks a game asset must pass, with numbers (NumPy + SciPy only)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def weld(verts: np.ndarray, faces: np.ndarray, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Merge vertices by position (UV / normal splits and separate layers share positions)."""
    q = np.round(np.asarray(verts, np.float64) / max(tol, 1e-9)).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return np.asarray(verts)[first], inv[faces]


def edge_stats(verts: np.ndarray, faces: np.ndarray, tol: float = 1e-4) -> Dict[str, int]:
    v, f = weld(verts, faces, tol)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return {"edges": int(len(counts)), "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()), "welded_vertices": int(len(v))}


def degenerate_faces(verts: np.ndarray, faces: np.ndarray, area_eps: float = 1e-12) -> int:
    p = np.asarray(verts, np.float64)
    a = 0.5 * np.linalg.norm(np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]]), axis=1)
    return int((a <= area_eps).sum())


def uv_report(uv: np.ndarray, faces: np.ndarray, size: Tuple[int, int] = (1024, 1024)) -> Dict[str, object]:
    """UV bounds, the share of the atlas that is covered, and the overlap factor (sum of UV triangle areas / covered area:
    1.0 = every texel belongs to one triangle, 2.0 = every texel is shared by two - shared UVs are intentional for the
    turned-copy sides of arenas and islands, but lightmap baking needs unique UVs)."""
    from .camera import rasterize_ortho
    w, h = size
    out: Dict[str, object] = {"min": [float(uv[:, 0].min()), float(uv[:, 1].min())],
                              "max": [float(uv[:, 0].max()), float(uv[:, 1].max())]}
    out["in_bounds"] = bool(uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6)
    xy = np.stack([uv[:, 0] * w, uv[:, 1] * h], 1)
    tid, _, _ = rasterize_ortho(xy, np.zeros(len(xy)), faces, h, w)
    covered = float((tid >= 0).sum())
    a, b, c = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
    area = float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])).sum())
    out["covered_fraction"] = covered / float(w * h)
    out["overlap_factor"] = area / max(covered, 1.0)
    return out


def texel_density(chart_scale: float, metres_per_px: float) -> float:
    """Texels per metre of surface seen by the reference camera."""
    return chart_scale / max(metres_per_px, 1e-12)
