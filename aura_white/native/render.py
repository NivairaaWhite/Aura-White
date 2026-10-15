"""Beauty renders of a (coloured) mesh for the native model: the pictures it is conditioned on and the pictures the
verifier compares against.  numpy + scipy only (re-uses Aura's exact orthographic z-buffer), so it runs anywhere.

Framing mirrors what is done to a user's picture at inference (`prep_picture`): crop to the object's bounding box, pad
to a square, object fills `fill` of the frame.  Training pictures and real pictures therefore look alike to the model.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..gen3d.views import ViewCam, render_maps

BG = 0.5                                   # neutral grey the RGB channels are composited over


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(n, 1e-12)


def fit_camera(verts: np.ndarray, az: float, el: float, size: int, fill: float = 0.86) -> ViewCam:
    cam0 = ViewCam(az, el, size)
    x0, x1, y0, y1 = cam0.extents(verts)
    ext = max(x1 - x0, y1 - y0, 1e-9)
    return ViewCam(az, el, size, fill * size / ext, (x0 + x1) / 2.0, (y0 + y1) / 2.0)


def view_dir(az: float, el: float) -> np.ndarray:
    a, e = np.radians(az), np.radians(el)
    return np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])


def render(verts: np.ndarray, faces: np.ndarray, colours: Optional[np.ndarray], az: float = 0.0, el: float = 0.0,
           size: int = 64, fill: float = 0.86, cam: Optional[ViewCam] = None, shade: bool = True):
    """-> (rgba float32 (S,S,4) with rgb composited over grey and alpha = mask, depth (S,S) NaN where empty, ViewCam).

    `colours` (V,3) in [0,1] or None (flat grey).  Lambert + ambient, light from the camera's upper left."""
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    cam = cam or fit_camera(v, az, el, size, fill)
    mask, depth, tri, bary = render_maps(v, f, cam)
    out = np.full((size, size, 4), BG, np.float32)
    out[..., 3] = 0.0
    if not mask.any():
        return out, depth, cam
    vn = vertex_normals(v, f)
    t = tri[mask]
    b = bary[mask].astype(np.float64)
    fv = f[t]                                                               # (K,3)
    n = (vn[fv] * b[..., None]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    c = np.full((len(t), 3), 0.72) if colours is None else (np.asarray(colours, np.float64)[fv] * b[..., None]).sum(1)
    if shade:
        d = view_dir(cam.az, cam.el)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, d)
        right /= max(np.linalg.norm(right), 1e-9)
        up_c = np.cross(d, right)
        L = d + 0.55 * up_c - 0.35 * right
        L /= np.linalg.norm(L)
        lam = np.clip(n @ L, 0.0, 1.0)
        rim = np.clip(1.0 - np.abs(n @ d), 0.0, 1.0) * 0.08
        c = c * (0.40 + 0.60 * lam[:, None]) + rim[:, None]
    out[mask, :3] = np.clip(c, 0.0, 1.0)
    out[mask, 3] = 1.0
    return out, depth, cam


def prep_picture(img: np.ndarray, size: int = 64, fill: float = 0.86) -> np.ndarray:
    """A user's picture (H,W,4 uint8/float, alpha = the cut-out) -> the model's input (S,S,4) float32: crop to the object,
    square pad, object fills `fill`, RGB composited over neutral grey (alpha kept as the 4th channel)."""
    from PIL import Image

    a = np.asarray(img)
    if a.dtype != np.float32 and a.dtype != np.float64:
        a = a.astype(np.float32) / 255.0
    if a.ndim == 2:
        a = np.stack([a, a, a, np.ones_like(a)], -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), a.dtype)], 2)
    m = a[..., 3] > 0.5
    if not m.any():
        raise ValueError("the picture has no foreground (alpha is empty) - cut the object out first")
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = a[y0:y1, x0:x1].copy()
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s, 4), np.float32)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    tgt = max(1, int(round(size * fill)))
    im = Image.fromarray((np.clip(sq, 0, 1) * 255).astype(np.uint8), "RGBA").resize((tgt, tgt), Image.LANCZOS)
    r = np.asarray(im).astype(np.float32) / 255.0
    out = np.zeros((size, size, 4), np.float32)
    o = (size - tgt) // 2
    out[o:o + tgt, o:o + tgt] = r
    alpha = out[..., 3:4]
    out[..., :3] = out[..., :3] * alpha + BG * (1.0 - alpha)
    out[..., 3] = (alpha[..., 0] > 0.5).astype(np.float32)
    return out


def to_uint8(rgba: np.ndarray) -> np.ndarray:
    return (np.clip(rgba, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def from_uint8(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) / 255.0


def contact_sheet(images: Sequence[np.ndarray], cols: int = 6, pad: int = 2) -> np.ndarray:
    """Tile (S,S,4) renders into one RGB uint8 picture (for reports)."""
    ims = [np.asarray(i) for i in images]
    s = ims[0].shape[0]
    rows = (len(ims) + cols - 1) // cols
    sheet = np.full((rows * (s + pad) + pad, cols * (s + pad) + pad, 3), 255, np.uint8)
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        rgb = im[..., :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        sheet[pad + r * (s + pad):pad + r * (s + pad) + s, pad + c * (s + pad):pad + c * (s + pad) + s] = rgb
    return sheet
