"""Silhouette snapping.

After registration a neural mesh matches the artwork silhouette to within a few pixels.  This pulls the outline of the
mesh onto the artwork outline (blade tips, fur tips, fingers) by moving silhouette vertices *within the image plane*
(their depth is kept), then relaxing the displacement over the mesh so the surface stays smooth.  Interior vertices are
moved only by the relaxation, never directly, so the reference view becomes silhouette-exact while the shape seen from
other angles changes as little as possible.  NumPy + SciPy only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import morph, softraster as sr


@dataclass
class SnapResult:
    verts: np.ndarray
    moved: int
    mean_shift_px: float
    iou_before: float
    iou_after: float


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def _laplacian(n: int, faces: np.ndarray):
    from scipy import sparse

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return A, np.maximum(deg, 1.0)


def snap_to_silhouette(verts: np.ndarray, faces: np.ndarray, cam, ref_alpha: np.ndarray, max_shift_px: float = 0.0,
                       relax: int = 12, depth_tol: float = 0.03, iterations: int = 2) -> SnapResult:
    """Pull the mesh's own silhouette onto `ref_alpha`'s silhouette as seen from `cam` (a single view). A thin
    wrapper over `snap_to_silhouettes([(cam, ref_alpha)], ...)` so the single- and multi-view code paths can never
    drift apart."""
    return snap_to_silhouettes(verts, faces, [(cam, ref_alpha)], max_shift_px=max_shift_px, relax=relax,
                               depth_tol=depth_tol, iterations=iterations)


def snap_to_silhouettes(verts: np.ndarray, faces: np.ndarray, views, max_shift_px: float = 0.0, relax: int = 12,
                        depth_tol: float = 0.03, iterations: int = 3) -> SnapResult:
    """Like `snap_to_silhouette`, but for several `(camera, ref_alpha)` views at once: every silhouette vertex is
    pulled towards *whichever* view(s) it is silhouette-visible in that iteration (its displacement is the average
    over those views) before the shared Laplacian relax runs once. A vertex on the mesh's true outline from every
    camera's vantage point gets a consistent pull from all of them; a vertex that is only ever an outline point in
    one view (e.g. a side profile a front view can't see) is still corrected by that one view. `views` is a sequence
    of `(camera, ref_alpha)` pairs sharing one output resolution (use the largest of the supplied alphas' shapes)."""
    from scipy import ndimage as ndi

    v = np.asarray(verts, np.float64).copy()
    f = np.asarray(faces, np.int64)
    A, deg = _laplacian(len(v), f)
    prepared = []
    for cam, ref_alpha in views:
        H, W = ref_alpha.shape
        ref = ref_alpha >= 0.5
        sd = ndi.gaussian_filter((morph.edt(ref) - morph.edt(~ref)).astype(np.float32), 1.0)
        gy, gx = np.gradient(sd)
        prepared.append((cam, ref, sd, gx, gy, H, W, max_shift_px or 0.02 * max(H, W)))
    ious_before = []
    for cam, ref, *_r, H, W, _ms in prepared:
        m0, _d, _t = sr.render_camera(cam, v, f, int(H), int(W))
        ious_before.append(_iou(m0, ref))
    moved_total = 0
    shift_sum = 0.0
    for _ in range(iterations):
        disp_sum = np.zeros_like(v)
        wgt_sum = np.zeros(len(v))
        any_sil = np.zeros(len(v), bool)
        for cam, ref, sd, gx, gy, H, W, max_shift in prepared:
            mask, depth, _t = sr.render_camera(cam, v, f, int(H), int(W))
            xy, w = sr.project(cam, v, int(H), int(W))
            ix = np.clip(np.floor(xy[:, 0]).astype(int), 0, W - 1)
            iy = np.clip(np.floor(xy[:, 1]).astype(int), 0, H - 1)
            inside_d = morph.edt(mask)
            outside_d = morph.edt(~mask)
            edge_d = np.where(mask, inside_d - 0.5, outside_d - 0.5)
            vis = np.abs(depth[iy, ix] - w) <= depth_tol * np.maximum(w, 1e-6)
            sil = (edge_d[iy, ix] <= 1.5) & vis & (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
            if not sil.any():
                continue
            s = ndi.map_coordinates(sd, [xy[sil, 1] - 0.5, xy[sil, 0] - 0.5], order=1, mode="nearest")
            gxs = ndi.map_coordinates(gx, [xy[sil, 1] - 0.5, xy[sil, 0] - 0.5], order=1, mode="nearest")
            gys = ndi.map_coordinates(gy, [xy[sil, 1] - 0.5, xy[sil, 0] - 0.5], order=1, mode="nearest")
            gn = np.maximum(np.hypot(gxs, gys), 1e-6)
            d_px = np.stack([gxs / gn, gys / gn], 1) * (-np.clip(s, -max_shift, max_shift))[:, None]
            target = xy[sil] + d_px
            newp = sr.unproject(cam, target[:, 0], target[:, 1], w[sil], int(H), int(W))
            idx = np.flatnonzero(sil)
            disp_sum[idx] += newp - v[idx]
            wgt_sum[idx] += 1.0
            any_sil[idx] = True
            shift_sum += float(np.abs(s).mean())
        if not any_sil.any():
            break
        disp = np.zeros_like(v)
        disp[any_sil] = disp_sum[any_sil] / wgt_sum[any_sil, None]
        d = disp.copy()
        for _r in range(relax):
            nd = (A @ d) / deg[:, None]
            d = np.where(any_sil[:, None], disp, 0.5 * d + 0.5 * nd)
        v = v + d
        moved_total += int(any_sil.sum())
    ious_after = []
    for cam, ref, *_r, H, W, _ms in prepared:
        m1, _d, _t = sr.render_camera(cam, v, f, int(H), int(W))
        ious_after.append(_iou(m1, ref))
    return SnapResult(v, moved_total, shift_sum / max(iterations, 1), float(np.mean(ious_before)),
                      float(np.mean(ious_after)))
