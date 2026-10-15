"""Is the mesh really three-dimensional?  (and is it a sound mesh at all)

Older code paths could hand back a *slab*: the front taken from the picture and the back its mirror image, a few
millimetres behind.  Judging such a result by how well its front matches the picture is meaningless - a slab matches the
front perfectly by construction.  This audit looks at what a front-view comparison cannot see:

  depth_ratio    depth / width of the bounding box
  thickness      per-pixel front-to-back distance over the silhouette (constant => extruded cut-out)
  mid-surface    the mean of the front and back surfaces (constant => the back is just a mirror of the front)
  flat_risk      0 (solid) .. 1 (relief-thin): 0 at the recipe's min_depth_ratio, 1 at half of it; only counts when the
                 recipe expects a solid object (a standing character is 0.4-0.8 as deep as wide, a relief slab 0.1-0.17)

plus topology: open edges, non-manifold edges, floating fragments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import meshutil as M
from .views import ViewCam, light_mesh, render_maps


@dataclass
class Audit:
    faces: int = 0
    watertight: bool = False
    boundary_edges: int = 0
    nonmanifold_edges: int = 0
    components: int = 0
    largest_share: float = 1.0
    extents: tuple = (0.0, 0.0, 0.0)
    depth_ratio: float = 1.0
    thickness_cv: float = 1.0
    mid_flatness: float = 1.0
    mirror_corr: float = 0.0
    profile: str = "solid"            # solid | thin | extruded | mirrored-relief
    flat_risk: float = 0.0
    topology_score: float = 1.0       # 1 = clean closed single shell
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["extents"] = [round(float(x), 5) for x in self.extents]
        for k, val in list(d.items()):
            if isinstance(val, (float, np.floating)):
                d[k] = round(float(val), 5)
        return d


def audit_mesh(verts: np.ndarray, faces: np.ndarray, min_depth_ratio: float = 0.22, expect_solid: bool = True,
               size: int = 160) -> Audit:
    from ..mesh.repair import topology_stats

    a = Audit()
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    a.faces = len(f)
    if len(f) < 4:
        a.notes.append("empty mesh")
        a.flat_risk, a.topology_score = 1.0, 0.0
        return a
    st = topology_stats(f)
    a.boundary_edges, a.nonmanifold_edges = int(st["boundary_edges"]), int(st["nonmanifold_edges"])
    a.watertight = a.boundary_edges == 0 and a.nonmanifold_edges == 0
    lab = M.component_labels(f, len(v))
    ids, counts = np.unique(lab, return_counts=True)
    a.components = int(len(ids))
    a.largest_share = float(counts.max() / counts.sum())
    ext = v.max(0) - v.min(0)
    a.extents = tuple(float(x) for x in ext)
    a.depth_ratio = float(ext[2] / max(ext[0], ext[1], 1e-9))
    # front / back depth maps with identical framing
    lv, lf = light_mesh(v, f)
    c = (lv.min(0) + lv.max(0)) / 2.0
    s = 0.92 * size / max(float(ext.max()), 1e-9)
    cam_f = ViewCam(0.0, 0.0, size, s, c[0], c[1])
    cam_b = ViewCam(180.0, 0.0, size, s, -c[0], c[1])
    mf, tf, _, _ = render_maps(lv, lf, cam_f)
    mb, tb, _, _ = render_maps(lv, lf, cam_b)
    mb, tb = mb[:, ::-1], tb[:, ::-1]                                 # back view is mirrored left-right
    both = mf & mb
    if both.sum() > 50:
        zf, zb = tf[both].astype(np.float64), -tb[both].astype(np.float64)       # z of the front / back surface
        thick = zf - zb
        mean_t = max(float(thick.mean()), 1e-9)
        a.thickness_cv = float(thick.std() / mean_t)
        a.mid_flatness = float(((zf + zb) / 2.0).std() / max(float(thick.max()), 1e-9))
        a.mirror_corr = float(np.corrcoef(zf, zb)[0, 1]) if zf.std() > 1e-9 and zb.std() > 1e-9 else 0.0
    thin = a.depth_ratio < min_depth_ratio
    if thin:
        if a.thickness_cv < 0.10 and a.mid_flatness < 0.10:
            a.profile = "extruded"
        elif a.mirror_corr < -0.85 or a.mid_flatness < 0.12:
            a.profile = "mirrored-relief"
        else:
            a.profile = "thin"
    a.flat_risk = float(np.clip((min_depth_ratio - a.depth_ratio) / max(0.5 * min_depth_ratio, 1e-9), 0.0, 1.0)) if expect_solid else 0.0
    if thin and expect_solid:
        a.notes.append(f"flat: depth is only {a.depth_ratio:.0%} of the width ({a.profile})")
    topo = 1.0
    if not a.watertight:
        topo -= 0.35 * min(1.0, (a.boundary_edges + 4 * a.nonmanifold_edges) / max(len(f) * 0.02, 1.0))
    topo -= 0.4 * (1.0 - a.largest_share)
    a.topology_score = float(np.clip(topo, 0.0, 1.0))
    if a.largest_share < 0.97:
        a.notes.append(f"{a.components} separate pieces ({(1 - a.largest_share):.1%} of the surface is loose fragments)")
    return a
