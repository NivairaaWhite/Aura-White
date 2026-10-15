"""Image loading, background removal and framing - Pillow + numpy only.

Background removal order (method="auto"):
  1. the picture already has transparency  -> used as-is (best quality, nothing to install)
  2. `rembg` if it happens to be installed and working (optional; it is the piece that most
     often breaks on new Python versions, so it is never required)
  3. built-in matte for plain / studio backgrounds (border-colour flood fill)
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps

MAX_SIDE = 2048
_rembg_session = None


@dataclass
class Prepared:
    rgb: np.ndarray           # (S,S,3) float32 in [0,1]: square, object on neutral gray - what the model sees
    cutout: Image.Image       # square RGBA cut-out (alpha = object mask)
    matting: str              # how the background was handled: alpha | rembg | border | none
    warnings: List[str] = field(default_factory=list)

    @property
    def preview(self) -> Image.Image:
        """8-bit picture of what the model sees (for saving / display)."""
        return Image.fromarray((np.clip(self.rgb, 0, 1) * 255.0).round().astype(np.uint8), "RGB")


def load_image(src, max_side: Optional[int] = None) -> Image.Image:
    """Accept a path, bytes, PIL image or numpy array; return an upright RGB/RGBA image.
    Pictures larger than `max_side` (default MAX_SIDE) are downscaled."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(src)))
    elif isinstance(src, np.ndarray):
        arr = src
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255.0).round().astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.open(Path(src))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        a = np.asarray(img, dtype=np.float32)
        a = a / max(float(a.max()), 1.0) * 255.0
        img = Image.fromarray(a.astype(np.uint8))
    if img.mode in ("P", "PA", "LA", "La", "RGBa") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    limit = max_side or MAX_SIDE
    if max(img.size) > limit:
        img = img.copy()
        img.thumbnail((limit, limit), Image.LANCZOS)
    img.load()
    return img


def _has_transparency(img: Image.Image) -> bool:
    if img.mode != "RGBA":
        return False
    lo, _ = img.getchannel("A").getextrema()
    return lo < 250


def _flood(seed: np.ndarray, allowed: np.ndarray, max_iter: int = 4000) -> np.ndarray:
    try:  # fast exact path if scipy is present (never required)
        from scipy import ndimage

        lab, _ = ndimage.label(allowed)
        ids = np.unique(lab[seed & allowed])
        return np.isin(lab, ids[ids > 0])
    except Exception:
        pass
    cur = seed & allowed
    count = int(cur.sum())
    for _ in range(max_iter):
        p = np.pad(cur, 1)
        nb = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | cur
        cur = nb & allowed
        n = int(cur.sum())
        if n == count:
            break
        count = n
    return cur


def _border_matte(rgb: Image.Image):
    """Alpha for objects photographed on a plain background. Returns (alpha L-image, note|None) or (None, why)."""
    work = rgb.copy()
    work.thumbnail((512, 512), Image.BILINEAR)
    a = np.asarray(work, dtype=np.float32)
    h, w, _ = a.shape
    b = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    d_border = np.linalg.norm(border - bg, axis=1)
    thr = float(np.clip(np.percentile(d_border, 95) * 1.6 + 8.0, 14.0, 70.0))
    dist = np.linalg.norm(a - bg, axis=2)
    seed = np.zeros((h, w), bool)
    seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = True
    bg_region = _flood(seed, dist <= thr)
    fg = ~bg_region
    frac = float(fg.mean())
    if frac < 0.005 or frac > 0.98:
        return None, "no clear foreground found against the border colour"
    note = None
    if float(np.mean(d_border <= thr)) < 0.85:
        note = "background looks busy/non-uniform - result may be rough; use a transparent PNG or install rembg"
    m = Image.fromarray((fg * 255).astype(np.uint8), "L")
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))  # drop specks
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.0))  # 1px inset removes colour halos
    return m.resize(rgb.size, Image.BICUBIC), note


def _try_rembg(img: Image.Image) -> Optional[Image.Image]:
    global _rembg_session
    try:
        import rembg  # optional dependency
    except Exception:
        return None
    try:
        if _rembg_session is None:
            _rembg_session = rembg.new_session()
        return rembg.remove(img.convert("RGB"), session=_rembg_session).convert("RGBA")
    except Exception:
        return None


def remove_background(img: Image.Image, method: str = "auto"):
    notes: List[str] = []
    if _has_transparency(img):
        return img, "alpha", notes
    rgb = img.convert("RGB")
    if method == "outline":
        from ..gen3d.outline import extract_outline
        o = extract_outline(rgb, strict=True)
        return o.rgba(), "outline", notes + list(o.notes)
    if method in ("auto", "exact"):
        # clean artwork on a flat background: a geometric matte keeps hairlines, thin blades and white-on-white shapes
        try:
            from .matte import plain_background_matte
            m = plain_background_matte(np.asarray(rgb))
        except Exception:
            m = None
        if m is not None:
            out = Image.fromarray(m.rgb).convert("RGBA")
            out.putalpha(Image.fromarray((np.clip(m.alpha, 0, 1) * 255 + 0.5).astype(np.uint8)))
            return out, "exact", notes + list(m.notes)
        if method == "exact":
            raise RuntimeError("the background is not a flat colour, so the exact matte cannot be used "
                               "(use --bg rembg or supply a transparent PNG)")
    if method in ("auto", "rembg"):
        out = _try_rembg(rgb)
        if out is not None:
            return out, "rembg", notes
        if method == "rembg":
            raise RuntimeError("rembg is not installed or failed to run (pip install rembg), or use --bg border")
    if method in ("auto", "border"):
        alpha, note = _border_matte(rgb)
        if alpha is not None:
            if note:
                notes.append(note)
            out = rgb.convert("RGBA")
            out.putalpha(alpha)
            return out, "border", notes
        notes.append(str(note) + " - using the whole picture as the object")
    rgba = rgb.convert("RGBA")
    return rgba, "none", notes


def frame_foreground(rgba: Image.Image, ratio: float = 0.85) -> Image.Image:
    """Crop to the object, centre it in a square canvas so it fills `ratio` of the side."""
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 12)
    if ys.size == 0:
        return rgba
    crop = rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    side = max(crop.size)
    canvas_side = int(round(side / max(min(ratio, 1.0), 0.05)))
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    canvas.paste(crop, ((canvas_side - crop.width) // 2, (canvas_side - crop.height) // 2))
    return canvas


def composite_gray(rgba: Image.Image, gray: float = 0.5) -> np.ndarray:
    a = np.asarray(rgba, dtype=np.float32) / 255.0
    return (a[..., :3] * a[..., 3:4] + (1.0 - a[..., 3:4]) * gray).astype(np.float32)


def prepare_image(src, remove_bg: bool = True, foreground_ratio: float = 0.85, matting: str = "auto",
                  max_side: Optional[int] = None) -> Prepared:
    img = load_image(src, max_side)
    warnings: List[str] = []
    if remove_bg:
        rgba, method, warnings = remove_background(img, matting)
        rgba = frame_foreground(rgba, foreground_ratio) if method != "none" else _square(rgba)
    else:
        rgba, method = _square(img.convert("RGBA")), "none"
    return Prepared(rgb=composite_gray(rgba), cutout=rgba, matting=method, warnings=warnings)


def _square(rgba: Image.Image) -> Image.Image:
    side = max(rgba.size)
    sq = Image.new("RGBA", (side, side), (127, 127, 127, 255))
    sq.paste(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))
    return sq
