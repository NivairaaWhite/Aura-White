"""Texture atlas: chart packing, gutters and the chart bakers (front / back / param)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .layers import Chart


def next_pot(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def pack_charts(charts: Sequence[Chart], max_size: int = 4096, padding: int = 8, pot: bool = True,
                min_size: int = 256) -> Tuple[int, int]:
    """Shelf-pack the charts (largest first).  Charts are scaled down uniformly until everything fits in
    `max_size`.  Writes `origin` and `scale` into each chart and returns the (width, height) of the atlas."""
    charts = list(charts)
    scale = 1.0
    for _ in range(40):
        order = sorted(charts, key=lambda c: -max(c.height, c.width))
        sizes = {c.cid: (max(4, int(round(c.width * scale * c.density))), max(4, int(round(c.height * scale * c.density))))
                 for c in order}
        # try a few atlas widths, keep the smallest area that fits
        best = None
        total_area = sum((w + 2 * padding) * (h + 2 * padding) for w, h in sizes.values())
        widest = max(w for w, _ in sizes.values()) + 2 * padding
        for W in sorted({min(max_size, max(widest, next_pot(int(np.sqrt(total_area) * k)))) for k in (1.0, 1.25, 1.6, 2.0)}):
            if W < widest or W > max_size:
                continue
            x = y = rowh = 0
            place = {}
            ok = True
            for c in order:
                w, h = sizes[c.cid]
                w += 2 * padding
                h += 2 * padding
                if x + w > W:
                    x, y, rowh = 0, y + rowh, 0
                place[c.cid] = (x + padding, y + padding)
                x += w
                rowh = max(rowh, h)
            H = y + rowh
            if H <= max_size:
                area = W * (next_pot(H) if pot else H)
                if best is None or area < best[0]:
                    best = (area, W, H, place)
        if best is not None:
            _, W, H, place = best
            for c in charts:
                c.origin = place[c.cid]
                c.scale = sizes[c.cid][0] / max(c.width, 1)   # includes the chart's density
            if pot:
                return next_pot(max(W, min_size)), next_pot(max(H, min_size))
            return W, max(H, min_size)
        scale *= 0.9
    raise RuntimeError("could not pack the texture charts")


def edge_pad(img: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")


def fill_outside(rgb: np.ndarray, alpha: np.ndarray, solid: float = 0.5) -> np.ndarray:
    """Replace everything outside the silhouette by the nearest object colour (bilinear sampling and mip-maps must
    never blend the old background back in)."""
    from scipy import ndimage as ndi
    ok = alpha >= solid
    if not ok.any() or ok.all():
        return rgb
    _, (iy, ix) = ndi.distance_transform_edt(~ok, return_distances=True, return_indices=True)
    return np.where(ok[..., None], rgb, rgb[iy, ix])


def soften(rgb: np.ndarray, alpha: np.ndarray, sigma: float, keep_rim: float) -> np.ndarray:
    """Detail-suppressed copy: strong blur inside, original colours kept within `keep_rim` px of the outline."""
    from scipy import ndimage as ndi
    m = (alpha >= 0.5).astype(np.float32)
    w = np.maximum(ndi.gaussian_filter(m, sigma), 1e-3)
    blur = np.stack([ndi.gaussian_filter(rgb[..., c].astype(np.float32) * m, sigma) / w for c in range(3)], -1)
    d = ndi.distance_transform_edt(m > 0)
    t = np.clip(d / max(keep_rim, 1.0), 0.0, 1.0)[..., None]
    t = t * t * (3 - 2 * t)
    return np.clip(rgb * (1 - t) + blur * t + 0.5, 0, 255).astype(np.uint8)


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return np.asarray(Image.fromarray(img).resize((w, h), Image.Resampling.LANCZOS))


def bake_front(src_filled: np.ndarray, ch: Chart) -> np.ndarray:
    x0, y0, x1, y1 = ch.crop
    tile = src_filled[y0:y1, x0:x1]
    return _resize(tile, max(4, int(round(ch.width * ch.scale))), max(4, int(round(ch.height * ch.scale))))


def bake_back(src_filled: np.ndarray, alpha: np.ndarray, ch: Chart, mode: str = "mirror",
              back_image: Optional[np.ndarray] = None, soft_sigma: float = 6.0, rim: float = 40.0) -> np.ndarray:
    x0, y0, x1, y1 = ch.crop
    if back_image is not None:
        tile = back_image[y0:y1, x0:x1][:, ::-1]                  # registered back picture, shown mirrored
    elif mode == "soft":
        tile = soften(src_filled, alpha, soft_sigma, rim)[y0:y1, x0:x1][:, ::-1]
    elif mode == "flat":
        a = (alpha >= 0.5)
        mean = src_filled[a].mean(0) if a.any() else np.array([128, 128, 128])
        tile = np.broadcast_to(mean.astype(np.uint8), src_filled[y0:y1, x0:x1].shape).copy()
        tile = soften(tile, alpha[y0:y1, x0:x1] * 0 + 1, 1.0, 1.0)
    else:
        tile = src_filled[y0:y1, x0:x1][:, ::-1]
    return _resize(np.ascontiguousarray(tile), max(4, int(round(ch.width * ch.scale))),
                   max(4, int(round(ch.height * ch.scale))))


def paste(atlas: np.ndarray, tile: np.ndarray, origin: Tuple[int, int], pad: int, wrap_u: bool = False) -> None:
    ox, oy = origin
    h, w = tile.shape[:2]
    if wrap_u:          # a chart that wraps around (angle): the gutter continues with the opposite edge
        padded = np.pad(tile, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        padded[:, :pad] = np.pad(tile, ((pad, pad), (0, 0), (0, 0)), mode="edge")[:, w - pad:w] if w >= pad else padded[:, :pad]
        padded[:, -pad:] = np.pad(tile, ((pad, pad), (0, 0), (0, 0)), mode="edge")[:, :pad] if w >= pad else padded[:, -pad:]
    else:
        padded = edge_pad(tile, pad)
    x0, y0 = ox - pad, oy - pad
    H, W = atlas.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex, ey = min(padded.shape[1], W - x0), min(padded.shape[0], H - y0)
    atlas[y0 + sy0:y0 + ey, x0 + sx0:x0 + ex] = padded[sy0:ey, sx0:ex]
