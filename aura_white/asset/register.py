"""Non-rigid alignment of a locked mesh to the picture (picture-plane warp, depth untouched).

A generated mesh matches the artwork's *outline* after `meshlock.fit_pose`, but its inner structure - the tops of the near
wall, towers, a sleeve - is usually off by tens of pixels.  Projected colours then land on the wrong surface and smear when
the model turns.  The structure the mesh *does* have (depth jumps, creases) shows up as edges in a depth/normal render, and
the artwork's structure shows up as edges in its luminance: a demons registration of the two edge maps gives a smooth
displacement field, applied to the vertices in the picture plane only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _resize(a: np.ndarray, shape) -> np.ndarray:
    from scipy import ndimage as ndi
    zy, zx = shape[0] / a.shape[-2], shape[1] / a.shape[-1]
    if a.ndim == 3:
        return np.stack([ndi.zoom(c, (zy, zx), order=1) for c in a])
    return ndi.zoom(a, (zy, zx), order=1)


def _warp(img: np.ndarray, u: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return ndi.map_coordinates(img, [yy + u[1], xx + u[0]], order=1, mode="nearest")


def demons(fixed: np.ndarray, moving: np.ndarray, levels=(8, 4, 2), iters=(80, 50, 30), sigma_fluid: float = 1.5,
           sigma_diff: float = 2.5, max_disp: float = 40.0) -> np.ndarray:
    """Displacement field u (2,H,W; x then y, pixels of the input size) with  moving(x + u(x)) ~ fixed(x)."""
    from scipy import ndimage as ndi
    H, W = fixed.shape
    u = np.zeros((2, H, W), np.float32)
    for lv, it in zip(levels, iters):
        h, w = max(16, H // lv), max(16, W // lv)
        F = _resize(fixed, (h, w)).astype(np.float32)
        M = _resize(moving, (h, w)).astype(np.float32)
        uu = _resize(u, (h, w)) / lv
        gy, gx = np.gradient(F)
        g2 = gx * gx + gy * gy
        for _ in range(it):
            Mw = _warp(M, uu)
            d = Mw - F
            den = g2 + d * d + 1e-6
            step = np.stack([-d * gx / den, -d * gy / den]).astype(np.float32)
            step = np.clip(step, -0.5, 0.5)
            step = np.stack([ndi.gaussian_filter(s, sigma_fluid) for s in step])
            uu = uu + step
            uu = np.stack([ndi.gaussian_filter(s, sigma_diff) for s in uu])
            uu = np.clip(uu, -max_disp / lv, max_disp / lv)
        u = _resize(uu * lv, (H, W)).astype(np.float32)
    return u


def edge_map(img: np.ndarray, sigma: float = 2.0, local: float = 25.0) -> np.ndarray:
    """Structure edges, contrast-normalised so a faint tower edge counts as much as a strong texture."""
    from scipy import ndimage as ndi
    g = ndi.gaussian_gradient_magnitude(img.astype(np.float32), 1.2)
    n = ndi.gaussian_filter(g, local) + 1e-3 * (float(g.max()) + 1e-6)
    return ndi.gaussian_filter(g / n, sigma).astype(np.float32)
