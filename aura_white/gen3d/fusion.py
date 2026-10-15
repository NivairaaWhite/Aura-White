"""Silhouette-consistent fusion: make a mesh agree with *every* supplied picture at once, in 3-D.

A generator sees its pictures through a neural network, so its mesh can drift from the exact outline - a missing horn,
a thinner arm, a tail that is cut short.  Image-plane tricks (snapping the outline of the front view) cannot repair that
without distorting the other views.  This module works on a voxel volume instead:

  carve   a voxel whose centre projects outside the silhouette of any view is removed
          (the visual hull is an upper bound on the true shape: nothing outside it can be right)
  grow    where a view shows object pixels the mesh does not cover, the volume is grown - one voxel layer at a time, from the
          existing surface, only inside the visual hull and only along rays that need it - so a dropped horn comes back
          without webbing the gaps elsewhere.  With a single view the growth is shallow (a thin plate), because one view
          says nothing about depth; with three or more views it may reach further.
  remesh  smoothed signed distance -> surface nets

The mesh is left untouched when it already agrees (so generator detail is never thrown away for nothing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import meshutil as M
from .views import ViewCam, ViewSpec, fit_cam, light_mesh, mask_iou, render_maps


@dataclass
class ViewConstraint:
    label: str
    cam: ViewCam                       # fitted to this picture
    mask: np.ndarray                   # bool (cam.size, cam.size)
    weight: float = 1.0


@dataclass
class FusionResult:
    verts: np.ndarray
    faces: np.ndarray
    applied: bool
    stats: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def build_constraints(verts: np.ndarray, faces: np.ndarray, views: Sequence[Tuple[ViewSpec, np.ndarray]],
                      size: int = 192) -> List[ViewConstraint]:
    """Fit one camera per (view, mask) to the mesh."""
    lv, lf = light_mesh(verts, faces)
    out = []
    for spec, mask in views:
        cam, _, tm = fit_cam(lv, lf, spec.az, spec.el, mask, size)
        out.append(ViewConstraint(spec.label, cam, tm))
    return out


def view_ious(verts: np.ndarray, faces: np.ndarray, constraints: Sequence[ViewConstraint]) -> Dict[str, float]:
    lv, lf = light_mesh(verts, faces)
    return {c.label: mask_iou(render_maps(lv, lf, c.cam)[0], c.mask) for c in constraints}


def _voxel_centres_slab(lo: np.ndarray, h: float, n: int, i0: int, i1: int) -> np.ndarray:
    ax = lo[0] + (np.arange(i0, i1) + 0.5) * h
    ay = lo[1] + (np.arange(n) + 0.5) * h
    az = lo[2] + (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def _sample(mask: np.ndarray, cam: ViewCam, P: np.ndarray) -> np.ndarray:
    x, y, _ = cam.project(P)
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ok = (xi >= 0) & (xi < cam.size) & (yi >= 0) & (yi < cam.size)
    out = np.zeros(len(P), bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    from scipy import ndimage as ndi
    return ndi.binary_dilation(mask, iterations=int(px))


def hull_and_need(constraints: Sequence[ViewConstraint], lo: np.ndarray, side: float, n: int,
                  uncovered: Dict[str, np.ndarray], tol_px: float):
    """(visual hull bool (n,n,n), needed bool (n,n,n)): `needed` marks voxels that project onto object pixels the mesh
    leaves uncovered in at least one view."""
    h = side / n
    hull = np.ones((n, n, n), bool)
    need = np.zeros((n, n, n), bool)
    slab = max(1, int(4_000_000 // (n * n)))
    for i0 in range(0, n, slab):
        i1 = min(n, i0 + slab)
        P = _voxel_centres_slab(lo, h, n, i0, i1)
        ok = np.ones(len(P), bool)
        nd = np.zeros(len(P), bool)
        for c in constraints:
            grow = _dilate(c.mask, int(np.ceil(tol_px * c.cam.size / 256.0)))
            ok &= _sample(grow, c.cam, P)
            u = uncovered.get(c.label)
            if u is not None and u.any():
                nd |= _sample(u, c.cam, P)
        hull[i0:i1] = ok.reshape(i1 - i0, n, n)
        need[i0:i1] = nd.reshape(i1 - i0, n, n)
    return hull, need


def fuse(verts: np.ndarray, faces: np.ndarray, constraints: Sequence[ViewConstraint], grid: int = 192,
         tol_px: float = 1.5, grow_iters: Optional[int] = None, min_change: float = 0.003,
         max_removed: float = 0.30, say=None) -> FusionResult:
    """Carve / grow `mesh` to agree with `constraints`.  Returns the original mesh unchanged when it already does."""
    say = say or (lambda *_: None)
    v, f = M.orient_outward(verts, faces)
    notes: List[str] = []
    if not constraints:
        return FusionResult(v, f, False, {}, ["no views to fuse with"])
    before = view_ious(v, f, constraints)
    lo, side = M.grid_for(v)
    n = int(grid)
    lv, lf = light_mesh(v, f, 60_000)
    occ = M.voxelize(lv, lf, n, lo, side)
    if occ.sum() < 8:
        return FusionResult(v, f, False, {"iou_before": float(np.mean(list(before.values())))}, ["mesh is empty or not closed"])
    uncovered = {}
    for c in constraints:
        sil = render_maps(lv, lf, c.cam)[0]
        uncovered[c.label] = _dilate(c.mask & ~_dilate(sil, 1), 0)
    hull, need = hull_and_need(constraints, lo, side, n, uncovered, tol_px)
    base = occ.copy()
    occ = occ & hull
    removed = float(base.sum() - occ.sum()) / max(float(base.sum()), 1.0)
    if removed > max_removed:
        # a mesh that is 70 % inside the pictures' silhouettes is far more likely mis-registered than wrong
        return FusionResult(v, f, False, {"iou_before": float(np.mean(list(before.values()))), "removed_fraction": removed},
                            [f"the pictures would remove {removed:.0%} of the volume - the views are probably not registered "
                             "to the mesh (wrong view labels?); kept the mesh unchanged"])
    # growth depth: one view says nothing about depth, several do
    spread = len({round(c.cam.az / 30.0) for c in constraints} | {round(c.cam.el / 30.0) * 100 for c in constraints})
    iters = grow_iters if grow_iters is not None else int(round(n / (10.0 if spread >= 3 else 32.0)))
    from scipy import ndimage as ndi
    struct = ndi.generate_binary_structure(3, 1)
    added_total = 0
    for _ in range(max(0, iters)):
        cand = ndi.binary_dilation(occ, structure=struct) & hull & need & ~occ
        k = int(cand.sum())
        if k == 0:
            break
        occ |= cand
        added_total += k
    added = added_total / max(float(base.sum()), 1.0)
    change = removed + added
    stats = {"iou_before": float(np.mean(list(before.values()))), "removed_fraction": removed, "added_fraction": added}
    if change < min_change:
        notes.append("the mesh already agrees with every view - left untouched")
        stats["iou_after"] = stats["iou_before"]
        return FusionResult(v, f, False, stats, notes)
    nv, nf = M.mesh_from_field(M.smooth_field(occ), lo, side)
    if len(nf) < 4:
        return FusionResult(v, f, False, stats, notes + ["fusion produced an empty mesh - ignored"])
    nv, nf = M.orient_outward(nv, nf)
    after = view_ious(nv, nf, constraints)
    stats["iou_after"] = float(np.mean(list(after.values())))
    stats["per_view_after"] = after
    if stats["iou_after"] + 1e-4 < stats["iou_before"]:
        notes.append("fusion would lower the agreement with the pictures - kept the original")
        return FusionResult(v, f, False, stats, notes)
    notes.append(f"carved {removed:.1%}, grew {added:.1%}; agreement {stats['iou_before']:.3f} -> {stats['iou_after']:.3f}")
    say(notes[-1])
    return FusionResult(nv, nf, True, stats, notes)
