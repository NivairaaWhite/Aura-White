"""Exact matte for artwork on a plain background (concept art, product shots, AI illustrations).

Learned matting models (rembg, U2-Net, ...) smooth away exactly the things this project cares about: hairline chains,
thin blades, fur tips, white-on-white silhouettes.  For clean illustrations on a flat background a geometric matte is
both sharper and more faithful, so it is the default; learned models remain available as a fallback or on request.

Steps
  1. the background colour is the median of the picture's border; a picture whose border is not (nearly) uniform is
     declined (returns None) so busier photographs are left to rembg / the old border matte
  2. "background-like" pixels are flood-filled from the border; the fill is *sealed* against 1-2 pixel gaps in an
     outline, so a broken contour line does not let the background leak into a white dress or a white tail
  3. background-like regions that are enclosed by the object are kept as holes only when they really are holes
     (surrounded by dark or saturated pixels, or very large and perfectly flat) - a white highlight inside a white star
     is not a hole
  4. the outer 1-px ring is defringed (light halo colours are replaced by the colour just inside)

Needs SciPy (already required by the studio extras); without it the caller falls back to the older matte.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..forge.morph import disk as _disk

try:                                   # SciPy is optional at import time; `available()` tells callers
    from scipy import ndimage as ndi
except Exception:                      # pragma: no cover
    ndi = None


def available() -> bool:
    return ndi is not None


@dataclass
class MatteResult:
    alpha: np.ndarray                  # (H,W) float32 in [0,1]; iso 0.5 is the sub-pixel silhouette
    rgb: np.ndarray                    # (H,W,3) uint8, defringed colours
    hard: np.ndarray                   # (H,W) bool object mask
    background: np.ndarray             # (3,) background colour
    tolerance: float
    holes: int                         # number of enclosed background regions kept as holes
    notes: List[str] = field(default_factory=list)


def plain_background_matte(rgb: np.ndarray, seal: float = 1.5, hole_min: int = 60, flood_window: int = 5, creep: int = 12,
                           tolerance: Optional[float] = None, min_uniform: float = 0.9) -> Optional[MatteResult]:
    """`rgb`: (H,W,3) uint8.  Returns None when the border is not a uniform colour."""
    if ndi is None:
        return None
    a = np.asarray(rgb[..., :3], dtype=np.float32)
    h, w, _ = a.shape
    b = max(2, int(round(0.012 * min(h, w))))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3),
                             a[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    d_border = np.linalg.norm(border - bg, axis=1)
    # p90, not p99: an object touching the picture edge must not inflate the estimate of the background noise
    tol = float(tolerance) if tolerance is not None else float(np.clip(np.percentile(d_border, 90) + 3.0, 5.0, 40.0))
    if float(np.mean(d_border <= tol)) < min_uniform:
        return None
    dist = np.linalg.norm(a - bg, axis=2)
    bright = dist <= tol
    if bright.mean() < 0.02 or bright.mean() > 0.995:
        return None
    notes: List[str] = []
    # A flat background has (almost) no local variation; pale fur / cloth that happens to be as bright as the
    # background does.  Flooding only through flat pixels stops the fill from leaking into white-on-white areas
    # whose outline is broken.
    gray = a.mean(axis=2)
    rng = ndi.maximum_filter(gray, size=5) - ndi.minimum_filter(gray, size=5)
    rb = float(np.median(np.concatenate([rng[:b].ravel(), rng[-b:].ravel(), rng[:, :b].ravel(), rng[:, -b:].ravel()])))
    flat_thr = float(np.clip(3.0 * rb + 1.0, 2.5, 12.0))
    flat = rng <= flat_thr
    rng3 = ndi.maximum_filter(gray, size=3) - ndi.minimum_filter(gray, size=3)
    flat3 = rng3 <= max(2.0, flat_thr * 0.7)         # finer test for small enclosed gaps
    # 5x5 flatness keeps the fill out of pale fur but also cannot enter the 6-10 px gaps between chains; the 3x3
    # test can.  Use the strict one unless the picture has hairline structures (many small flat islands).
    flood_flat = flat3 if flood_window == 3 else flat

    # 2. sealed flood fill from the border
    s = _disk(seal)
    core = ndi.binary_erosion(bright & flood_flat, s, border_value=1)
    lab, _ = ndi.label(core)
    seeds = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    seeds = seeds[seeds > 0]
    bg_region = np.isin(lab, seeds)
    # gaps between hairlines (chains, wires) are too narrow for the 5x5 test: let the fill creep a little further
    # through 3x3-flat pixels only; a broken outline can therefore leak at most `creep` px into pale fur
    if flood_window != 3 and creep > 0:
        bg_region = ndi.binary_dilation(bg_region, iterations=int(creep), mask=bright & flat3)
    # the 2-3 px halo next to an outline is not flat either, but it is background: grow back into `bright`
    bg_region = ndi.binary_dilation(bg_region, _disk(3.0)) & bright
    # pixels of `bright` that the sealed fill did not reach but which touch the flooded region without crossing a
    # real outline (the 1-2 px rim removed by the erosion) belong to the background as well
    obj = ~bg_region

    # 3. enclosed background-like regions: holes or highlights?  Only the FLAT interior of a region is tested (its
    #    halo next to an outline is not flat and would glue neighbouring regions together).
    enclosed = bright & flat3 & obj
    lab2, n2 = ndi.label(enclosed)
    holes = 0
    if n2:
        area = np.bincount(lab2.ravel(), minlength=n2 + 1)
        big = 0.005 * h * w
        slices = ndi.find_objects(lab2)
        for i in np.flatnonzero(area[1:] >= hole_min) + 1:
            sl = slices[i - 1]
            pad = 9
            y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, h)
            x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, w)
            comp = lab2[y0:y1, x0:x1] == i
            near = ndi.binary_dilation(comp, _disk(3.0))
            # what surrounds the region: outline pixels only (the bright halo pixels are skipped)
            ring = ndi.binary_dilation(comp, _disk(7.0)) & ~comp & (dist[y0:y1, x0:x1] >= 30.0)
            if int(ring.sum()) < 6:
                continue
            dark_frac = float(np.mean(dist[y0:y1, x0:x1][ring] >= 90.0))
            if dark_frac >= 0.6 or (area[i] >= big and dark_frac >= 0.25):
                obj[y0:y1, x0:x1][near & bright[y0:y1, x0:x1]] = False
                holes += 1
    hard = obj

    # tidy: single-pixel specks on both sides
    lab3, n3 = ndi.label(hard, structure=np.ones((3, 3)))
    if n3:
        sizes = np.bincount(lab3.ravel(), minlength=n3 + 1)
        small = sizes < 6
        small[0] = False
        if small.any():
            hard = hard & ~small[lab3]
    frac = float(hard.mean())
    if frac < 0.003 or frac > 0.98:
        return None

    # 4. defringe the outer ring: keep whichever of (pixel, colour just inside) is farther from the background
    inner1 = ndi.binary_erosion(hard, _disk(1.5))
    inner0 = ndi.binary_erosion(hard, _disk(0.9))
    src = inner1 | (inner0 & ~ndi.binary_dilation(inner1, _disk(2.5)))
    rim = hard & ~inner1
    out = a.copy()
    if src.any() and rim.any():
        iy, ix = ndi.distance_transform_edt(~src, return_distances=False, return_indices=True)
        near = a[iy, ix]
        d_here = dist
        d_near = np.linalg.norm(near - bg, axis=2)
        swap = rim & (d_near > d_here)
        out[swap] = near[swap]
    alpha = ndi.gaussian_filter(hard.astype(np.float32), 0.7)
    alpha = np.where(hard, np.maximum(alpha, 0.5), np.minimum(alpha, 0.49)).astype(np.float32)
    if holes:
        notes.append(f"matte: {holes} enclosed background region(s) kept as holes")
    if float(np.mean(d_border <= tol)) < 0.97:
        notes.append("matte: background is slightly non-uniform; check the *_matte.png preview")
    return MatteResult(alpha, np.clip(out + 0.5, 0, 255).astype(np.uint8), hard, bg, tol, holes, notes)
