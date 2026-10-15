"""NumPy preview renderer: textured turntables, reference-view comparison, contact sheets."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..forge.softraster import project, rasterize
from ..raster.camera import Camera, fov_to_f, look_at


def _sample_bilinear(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = tex.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1.001)
    y = np.clip(uv[:, 1] * h - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    t = tex.astype(np.float32)
    return ((t[y0, x0] * (1 - fx) + t[y0, x0 + 1] * fx) * (1 - fy)
            + (t[y0 + 1, x0] * (1 - fx) + t[y0 + 1, x0 + 1] * fx) * fy)


def render_view(verts, faces, uv, normals, texture, eye, target=(0, 0, 0), size=(512, 512), fov=24.0,
                shade: str = "light", bg=(34, 36, 44), up=(0, 1, 0), emissive: Optional[np.ndarray] = None,
                cull: bool = False):
    """Render the textured mesh from `eye`.  Returns (rgb uint8 (H,W,3), coverage bool (H,W))."""
    Wd, Hd = size
    cam = Camera(look_at(eye, target, up), fov_to_f(fov))
    xy, w = project(cam, np.asarray(verts, np.float64), Hd, Wd)
    F = np.asarray(faces, np.int64)
    if cull:
        p = xy[F]
        area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
        F = F[area < 0]
    tid, bary, _ = rasterize(xy, w, F, Hd, Wd)
    hit = tid >= 0
    out = np.empty((Hd, Wd, 3), np.float32)
    out[:] = np.array(bg, np.float32)
    if not hit.any():
        return out.astype(np.uint8), hit
    t = tid[hit]
    b = bary[hit]
    tri = F[t]
    uvp = (uv[tri] * b[..., None]).sum(1)
    col = _sample_bilinear(texture, uvp)
    if shade == "light":
        nrm = (normals[tri] * b[..., None]).sum(1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        view = np.asarray(eye, np.float64) - np.asarray(target, np.float64)
        view /= np.linalg.norm(view)
        # headlight + a fixed key light, double sided
        key = np.array([0.4, 0.7, 0.6])
        key /= np.linalg.norm(key)
        ndv = np.abs(nrm @ view)
        ndk = np.abs(nrm @ key)
        col = col * (0.55 + 0.30 * ndv + 0.25 * ndk)[:, None]
    out[hit] = np.clip(col, 0, 255)
    return out.astype(np.uint8), hit


def turntable(verts, faces, uv, normals, texture, azimuths: Sequence[float] = (0, 45, 90, 135, 180, 225, 270, 315),
              elevation: float = 12.0, tile: int = 360, fov: float = 24.0, shade: str = "light",
              labels: bool = True, extra_views: Sequence[Tuple[str, float, float]] = ()) -> np.ndarray:
    """Contact sheet.  0 deg = the reference side (+z); positive angles orbit to the object's right."""
    v = np.asarray(verts, np.float64)
    lo, hi = v.min(0), v.max(0)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2
    dist = radius / np.tan(np.radians(fov) / 2) * 1.08
    tiles = []
    views = [(f"{int(a)}°", a, elevation) for a in azimuths] + list(extra_views)
    for label, az, el in views:
        a, e = np.radians(az), np.radians(el)
        eye = centre + dist * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        img, _ = render_view(v, faces, uv, normals, texture, eye, centre, (tile, tile), fov, shade)
        if labels:
            im = Image.fromarray(img)
            ImageDraw.Draw(im).text((6, 4), label, fill=(255, 255, 255))
            img = np.asarray(im)
        tiles.append(img)
    cols = 4 if len(tiles) > 4 else len(tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
    return sheet


def render_reference(cam, verts, faces, uv, texture, scale: float = 1.0):
    """Render through the *orthographic reference camera* (what the artwork was drawn with).  Returns
    (rgb uint8 (H,W,3), coverage bool (H,W), depth t (H,W))."""
    from .camera import rasterize_ortho
    H, W = int(round(cam.height * scale)), int(round(cam.width * scale))
    sx, sy, t = cam.project(verts)
    tid, bary, depth = rasterize_ortho(np.stack([sx * scale, sy * scale], 1), t, faces, H, W, nearest_is_larger=True)
    hit = tid >= 0
    out = np.zeros((H, W, 3), np.uint8)
    if hit.any():
        tri = np.asarray(faces)[tid[hit]]
        uvp = (uv[tri] * bary[hit][..., None]).sum(1)
        out[hit] = np.clip(_sample_bilinear(texture, uvp) + 0.5, 0, 255).astype(np.uint8)
    return out, hit, depth


def fidelity(rgb_ref: np.ndarray, alpha_ref: np.ndarray, rendered: np.ndarray, hit: np.ndarray):
    """Silhouette IoU and colour PSNR (inside the union that both agree is object) of a reference-camera render."""
    mask = alpha_ref >= 0.5
    iou = float((mask & hit).sum() / max((mask | hit).sum(), 1))
    both = mask & hit
    if both.any():
        mse = float(((rendered.astype(np.float64) - rgb_ref.astype(np.float64))[both] ** 2).mean())
        psnr = 99.0 if mse < 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)
    else:
        psnr = 0.0
    return iou, psnr


def fidelity_report(rgb_ref: np.ndarray, alpha_ref: np.ndarray, rendered: np.ndarray, hit: np.ndarray, tol: int = 2):
    """Reference-view agreement in three numbers: silhouette IoU, `outline_err` (share of the matte's pixels that are wrong by
    more than `tol` px - thin parts make IoU harsh, a 1 px wobble on a shaft is not an error) and `psnr_core`
    (colour PSNR away from the outline, where the picture should be reproduced exactly)."""
    from scipy import ndimage as ndi
    mask = alpha_ref >= 0.5
    iou, psnr = fidelity(rgb_ref, alpha_ref, rendered, hit)
    near_m = ndi.binary_dilation(mask, iterations=tol) if tol else mask
    near_h = ndi.binary_dilation(hit, iterations=tol) if tol else hit
    err = (hit & ~near_m) | (mask & ~near_h)
    outline_err = float(err.sum() / max(mask.sum(), 1))
    core = ndi.binary_erosion(mask & hit, iterations=3)
    if core.any():
        mse = float(((rendered.astype(np.float64) - rgb_ref.astype(np.float64))[core] ** 2).mean())
        psnr_core = 99.0 if mse < 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)
    else:
        psnr_core = psnr
    return dict(iou=iou, psnr=psnr, outline_err=outline_err, psnr_core=float(min(psnr_core, 99.0)))
