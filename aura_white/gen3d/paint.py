"""Projecting extra pictures of the object (user photos of the back and sides, or synthesised views) into the texture atlas.

The reference picture colours what the reference camera sees, 1:1.  Everything else used to be a smooth harmonic fill -
which is why a back looked like a blurry echo of the front.  Here every extra view is registered to the mesh (its camera
is *fitted* by silhouette, so photographs and synthesised views are both fine), each atlas texel is traced back to its 3-D
surface point, and the view contributes its colour where it really sees that point:

  * visible      the texel's depth matches the view's z-buffer (nothing in front of it)
  * facing       weight ~ cos(angle)^power - grazing views contribute little
  * silhouette   weight fades near the view's outline, where colour bleeds from the background
  * reference    texels the reference picture sees keep (almost all of) their colour; unseen texels yield to the views

Pure numpy; no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .views import ViewCam, fit_cam, light_mesh, mask_iou, render_maps


@dataclass
class ExtraView:
    label: str
    az: float                          # degrees, relative to the reference view (0 = the picture itself)
    el: float
    image: np.ndarray                  # (S, S, 3) uint8, square
    mask: np.ndarray                   # (S, S) bool
    weight: float = 1.0                # confidence: photos 1.0, synthesised views lower
    source: str = "user"               # user | synthesised


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(x - 0.5, 0, w - 1.001)
    y = np.clip(y - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    a = img[y0, x0].astype(np.float32)
    b = img[y0, x0 + 1].astype(np.float32)
    c = img[y0 + 1, x0].astype(np.float32)
    d = img[y0 + 1, x0 + 1].astype(np.float32)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def texel_surface(verts: np.ndarray, faces: np.ndarray, uv: np.ndarray, hw: Tuple[int, int]):
    """For every atlas texel: (covered bool (H,W), 3-D point (H,W,3), face normal (H,W,3))."""
    from ..asset.camera import rasterize_ortho

    H, W = hw
    xy = np.stack([np.asarray(uv, np.float64)[:, 0] * W, np.asarray(uv, np.float64)[:, 1] * H], 1)
    tri, bary, _ = rasterize_ortho(xy, np.zeros(len(xy)), np.asarray(faces, np.int64), H, W, nearest_is_larger=True)
    cov = tri >= 0
    P = np.zeros((H, W, 3))
    N = np.zeros((H, W, 3))
    if cov.any():
        t = tri[cov]
        fv = verts[faces[t]]                                              # (n, 3, 3)
        P[cov] = np.einsum("nk,nkj->nj", bary[cov], fv)
        fn = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
        N[cov] = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    return cov, P, N


def _visibility(project: Callable, size_hw: Tuple[int, int], verts: np.ndarray, faces: np.ndarray, P: np.ndarray,
                eps: float):
    """(visible bool (n,), x_px, y_px) of surface points P seen by a camera with `project(P) -> (x, y, t)`."""
    from ..asset.camera import rasterize_ortho

    h, w = size_hw
    sx, sy, st = project(verts)
    _, _, depth = rasterize_ortho(np.stack([sx, sy], 1), st, np.asarray(faces, np.int64), h, w, nearest_is_larger=True)
    x, y, t = project(P)
    xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    inb = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    vis = np.zeros(len(P), bool)
    tm = np.full(len(P), np.nan)
    tm[inb] = depth[yi[inb], xi[inb]]
    vis[inb] = np.isfinite(tm[inb]) & (t[inb] >= tm[inb] - eps)
    return vis, x, y


def camera_dir(az: float, el: float) -> np.ndarray:
    a, e = np.radians(az), np.radians(el)
    return np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])


def register_view(verts: np.ndarray, faces: np.ndarray, view: ExtraView, size: int = 192,
                  az_search: Sequence[float] = (0.0, -12.0, 12.0, -24.0, 24.0)) -> Tuple[ViewCam, float]:
    """Fit the camera of an extra view to the mesh (silhouette IoU), allowing a small azimuth correction."""
    lv, lf = light_mesh(verts, faces)
    best = None
    for da in az_search:
        cam, iou, _ = fit_cam(lv, lf, view.az + da, view.el, view.mask, size, refine=False)
        if best is None or iou > best[1]:
            best = (cam, iou)
    cam, iou, _ = fit_cam(lv, lf, best[0].az, view.el, view.mask, size, refine=True)
    return (cam, iou) if iou >= best[1] else best


def project_views_into_atlas(atlas: np.ndarray, mesh: Tuple[np.ndarray, np.ndarray, np.ndarray], views: Sequence[ExtraView],
                             ref_project: Optional[Callable] = None, ref_size_hw: Optional[Tuple[int, int]] = None,
                             power: float = 2.0, min_facing: float = 0.2, seen_keep: float = 6.0, unseen_keep: float = 0.05,
                             min_iou: float = 0.55, gutter: int = 8, log: Optional[Callable[[str], None]] = None):
    """Blend `views` into the albedo `atlas` (H,W,3 uint8).  `mesh` = (verts, faces, uv) of the atlas' LOD0 mesh.
    Returns (new atlas uint8, report dict)."""
    log = log or (lambda *_: None)
    verts, faces, uv = (np.asarray(mesh[0], np.float64), np.asarray(mesh[1], np.int64), np.asarray(mesh[2], np.float64))
    H, W = atlas.shape[:2]
    cov, P3, N3 = texel_surface(verts, faces, uv, (H, W))
    idx = np.flatnonzero(cov.ravel())
    P, N = P3.reshape(-1, 3)[idx], N3.reshape(-1, 3)[idx]
    extent = float(np.ptp(verts, axis=0).max())
    eps = 0.012 * extent
    base_w = np.full(len(idx), unseen_keep, np.float32)
    if ref_project is not None and ref_size_hw is not None:
        seen, _, _ = _visibility(ref_project, ref_size_hw, verts, faces, P, eps)
        base_w[seen] = seen_keep
    acc = np.zeros((len(idx), 3), np.float32)
    wsum = np.zeros(len(idx), np.float32)
    report: Dict[str, object] = {"views": []}
    from scipy import ndimage as ndi
    for v in views:
        cam, iou = register_view(verts, faces, v)
        info = {"label": v.label, "source": v.source, "az": round(cam.az, 1), "registration_iou": round(float(iou), 3)}
        if iou < min_iou:
            info["skipped"] = f"does not line up with the mesh (silhouette IoU {iou:.2f} < {min_iou})"
            log(f"  view '{v.label}': {info['skipped']}")
            report["views"].append(info)
            continue
        S = v.image.shape[0]
        k = S / float(cam.size)
        hi = cam.with_(size=S, scale=cam.scale * k)
        vis, x, y = _visibility(hi.project, (S, S), verts, faces, P, eps)
        facing = N @ camera_dir(cam.az, cam.el)
        edge = ndi.distance_transform_edt(np.pad(v.mask, 1))[1:-1, 1:-1]
        xi, yi = np.clip(np.floor(x).astype(np.int64), 0, S - 1), np.clip(np.floor(y).astype(np.int64), 0, S - 1)
        w = vis & (facing > min_facing)
        wt = np.where(w, (np.clip(facing, 0, 1) ** power) * _smoothstep(edge[yi, xi] / max(0.012 * S, 2.0)) * v.weight, 0.0).astype(np.float32)
        use = wt > 1e-4
        if use.any():
            col = _bilinear(v.image, x[use], y[use])
            acc[use] += col * wt[use, None]
            wsum[use] += wt[use]
        info["texels"] = int(use.sum())
        report["views"].append(info)
        log(f"  view '{v.label}': registered at {cam.az:.0f} deg (IoU {iou:.2f}), colours {int(use.sum()):,} texels")
    out = atlas.astype(np.float32).reshape(-1, 3).copy()
    flat_idx = idx
    base = out[flat_idx]
    new = np.where((wsum > 0)[:, None], (base * base_w[:, None] + acc) / (base_w + wsum)[:, None], base)
    out[flat_idx] = new
    out = out.reshape(H, W, 3)
    # re-grow the gutters from the updated islands so seams do not show colours from before the update
    if gutter > 0 and cov.any():
        d, (iy, ix) = ndi.distance_transform_edt(~cov, return_indices=True)
        ring = (~cov) & (d <= gutter)
        out[ring] = out[iy[ring], ix[ring]]
    report["updated_fraction"] = float((wsum > 0).mean()) if len(wsum) else 0.0
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), report
