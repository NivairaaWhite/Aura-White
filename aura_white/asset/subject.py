"""Load the artwork, cut the subject out, and decide what kind of thing it is.

Matting order (``matte="auto"``):
  alpha   the picture already carries transparency                         (used exactly as given)
  file    ``matte_file=`` - your own cut-out / black-white mask            (always wins when given)
  plain   flat-colour background: Aura's exact geometric matte             (keeps hairlines, white-on-white shapes)
  disc    a round object on a dark / uniform background (moons, orbs, planets, coins): a least-squares circle or
          ellipse - the sphere builder needs a *perfect* silhouette and not the 3-px ragged edge of an AI render
  rembg   busy backgrounds (sky, clouds, interiors) - u2net / isnet through ``rembg`` (optional install)
  border  last resort: flood fill from the picture border

Subject kinds drive the 3D proxy (see ``proxies.py``):
  sphere     a ball                                  -> true UV sphere, hemisphere mirrored to the far side
  character  a standing figure                       -> rounded relief with a lens-shaped body section
  prop       thin, long, hand-held / floor objects   -> rounded relief with round cords and thin blades
  arena      a ring structure seen from above        -> body of revolution (outer wall, stands, floor)
  island     a floating island with buildings        -> revolved rock + plateau + upright relief for the top
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

KINDS = ("sphere", "character", "prop", "arena", "island", "building")


@dataclass
class Subject:
    rgb: np.ndarray                    # (H,W,3) uint8 source colours
    alpha: np.ndarray                  # (H,W) float32; the 0.5 iso-line is the silhouette
    matte: str                         # which matte produced `alpha`
    notes: List[str] = field(default_factory=list)
    kind: str = "character"
    kind_reason: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    disc: Optional[Tuple[float, float, float, float]] = None   # (cx, cy, rx, ry) in px when the subject is round
    name: str = "asset"

    @property
    def hard(self) -> np.ndarray:
        return self.alpha >= 0.5

    @property
    def size(self) -> Tuple[int, int]:        # (width, height)
        return self.rgb.shape[1], self.rgb.shape[0]


# ------------------------------------------------------------------------------------------------ loading
def load_rgba(src, max_side: int = 2048) -> Image.Image:
    from ..image.preprocess import load_image
    return load_image(src, max_side)


def _ndi():
    from scipy import ndimage
    return ndimage


# ------------------------------------------------------------------------------------------------ mattes
def _fit_axis_ellipse(ex: np.ndarray, ey: np.ndarray):
    """Algebraic fit of  ((x-cx)/rx)^2 + ((y-cy)/ry)^2 = 1  on centred, unit-scaled coordinates (well conditioned)."""
    mx, my = float(ex.mean()), float(ey.mean())
    sc = float(np.hypot(ex - mx, ey - my).mean()) or 1.0
    u, v = (ex - mx) / sc, (ey - my) / sc
    M = np.c_[u ** 2, v ** 2, u, v]
    p = np.linalg.lstsq(M, np.ones(len(u)), rcond=None)[0]      # p0 u^2 + p1 v^2 + p2 u + p3 v = 1
    if p[0] <= 0 or p[1] <= 0:
        return None
    cu, cv = -p[2] / (2 * p[0]), -p[3] / (2 * p[1])
    rhs = 1.0 + p[0] * cu ** 2 + p[1] * cv ** 2
    ru, rv = np.sqrt(rhs / p[0]), np.sqrt(rhs / p[1])
    return mx + cu * sc, my + cv * sc, ru * sc, rv * sc


def disc_matte(rgb: np.ndarray, min_iou: float = 0.97):
    """A round object on a dark/uniform background -> (alpha, (cx, cy, rx, ry), iou) or None."""
    ndi = _ndi()
    a = rgb[..., :3].astype(np.float32)
    h, w, _ = a.shape
    b = max(3, int(round(0.015 * min(h, w))))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3),
                             a[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(a - bg, axis=2)
    spread = float(np.percentile(np.linalg.norm(border - bg, axis=1), 95))
    best = None
    for thr in (24.0, 34.0, 18.0, 48.0):
        fg = dist > max(thr, spread * 1.1)
        fg = ndi.binary_opening(fg, structure=np.ones((5, 5), bool))          # stars, sparkles, noise
        lab, n = ndi.label(fg)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        k = int(sizes.argmax())
        if sizes[k] < 0.02 * h * w:
            continue
        m = ndi.binary_fill_holes(ndi.binary_closing(lab == k, structure=np.ones((3, 3), bool), iterations=7))
        edge = m & ~ndi.binary_erosion(m)
        ey, ex = np.nonzero(edge)
        if len(ex) < 40:
            continue
        fit = _fit_axis_ellipse(ex.astype(np.float64) + 0.5, ey.astype(np.float64) + 0.5)
        if fit is None:
            continue
        cx, cy, rx, ry = fit
        if not (np.isfinite(fit).all() and 0.5 < rx / ry < 2.0 and rx > 4 and ry > 4):
            continue
        yy, xx = np.mgrid[0:h, 0:w]
        q = np.sqrt(((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2)
        ins = q <= 1.0
        iou = float((ins & m).sum() / max((ins | m).sum(), 1))
        if best is None or iou > best[0]:
            best = (iou, fit, q, 0.5 * (rx + ry))
    if best is None or best[0] < min_iou:
        return None
    iou, (cx, cy, rx, ry), q, rm = best
    sd = (1.0 - q) * rm                                                   # signed distance to the rim in px
    alpha = np.clip(sd + 0.5, 0.0, 1.0).astype(np.float32)               # one-pixel anti-aliased edge
    return alpha, (float(cx), float(cy), float(rx), float(ry)), iou


def _rembg_matte(img: Image.Image, model: str = "u2net"):
    try:
        import rembg
    except Exception:
        return None
    try:
        sess = rembg.new_session(model)
        out = rembg.remove(img.convert("RGB"), session=sess, alpha_matting=False)
        return np.asarray(out.convert("RGBA"))[..., 3].astype(np.float32) / 255.0
    except Exception:
        return None


def _largest_blobs(hard: np.ndarray, keep_ratio: float = 0.004) -> np.ndarray:
    """Drop floating specks that are tiny next to the main object (matting noise)."""
    ndi = _ndi()
    lab, n = ndi.label(hard, structure=np.ones((3, 3)))
    if n <= 1:
        return hard
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes >= max(8, keep_ratio * sizes.max())
    keep[0] = False
    return keep[lab]


def make_alpha(img: Image.Image, matte: str = "auto", matte_file=None, rembg_model: str = "u2net"):
    """Returns (rgb uint8, alpha float32, method, notes, disc|None)."""
    notes: List[str] = []
    rgba = np.asarray(img.convert("RGBA"))
    rgb = rgba[..., :3].copy()
    has_alpha = img.mode == "RGBA" and float(rgba[..., 3].min()) < 250

    if matte_file is not None or matte == "file":
        if matte_file is None:
            raise ValueError("matte='file' needs matte_file=...")
        m = Image.open(matte_file)
        m = m.getchannel("A") if m.mode in ("RGBA", "LA") and m.getextrema()[-1][0] < 250 else m.convert("L")
        if m.size != img.size:
            m = m.resize(img.size, Image.Resampling.BILINEAR)
        return rgb, np.asarray(m, np.float32) / 255.0, "file", notes, None
    if matte in ("auto", "alpha") and has_alpha:
        a = rgba[..., 3].astype(np.float32) / 255.0
        return rgb, a, "alpha", notes, None
    flat = img.convert("RGB")
    if matte in ("auto", "plain", "exact"):
        try:
            from ..image.matte import plain_background_matte
            m = plain_background_matte(np.asarray(flat))
        except Exception:
            m = None
        if m is not None:
            # a round object on a flat dark background is better served by the disc fit
            if matte == "auto":
                d = disc_matte(np.asarray(flat))
                if d is not None:
                    notes.append(f"matte: round subject (ellipse fit IoU {d[2]:.4f})")
                    return rgb, d[0], "disc", notes, d[1]
            notes += list(m.notes)
            return np.asarray(m.rgb), m.alpha.astype(np.float32), "plain", notes, None
        if matte in ("plain", "exact"):
            raise RuntimeError("the background is not a flat colour; use matte='rembg' or supply matte_file")
    if matte in ("auto", "disc"):
        d = disc_matte(np.asarray(flat))
        if d is not None:
            notes.append(f"matte: round subject (ellipse fit IoU {d[2]:.4f})")
            return rgb, d[0], "disc", notes, d[1]
        if matte == "disc":
            raise RuntimeError("no round subject found")
    if matte in ("auto", "rembg"):
        a = _rembg_matte(flat, rembg_model)
        if a is not None:
            hard = _largest_blobs(a >= 0.5)
            a = np.where(hard, np.maximum(a, 0.5), np.minimum(a, 0.49)).astype(np.float32)
            notes.append(f"matte: rembg/{rembg_model}")
            return rgb, a, "rembg", notes, None
        if matte == "rembg":
            raise RuntimeError("rembg is not installed or failed (pip install rembg onnxruntime)")
    if matte in ("auto", "outline"):
        # Aura White 7: a scored multi-method outline (isnet-anime / birefnet / grabcut / line-art fill ...) instead of
        # a bare border flood fill or - worse - the whole picture, which is what turns a character into a pasted slab.
        from ..gen3d.outline import OutlineError, extract_outline
        try:
            o = extract_outline(flat, strict=(matte == "outline"))
        except OutlineError:
            raise
        except Exception as e:                                         # never let an optional extra break the pipeline
            o = None
            notes.append(f"matte: outline extractor unavailable ({type(e).__name__})")
        if o is not None and o.method != "none":
            notes += list(o.notes)
            return np.asarray(o.rgb), o.alpha.astype(np.float32), "outline", notes, None
    from ..image.preprocess import _border_matte
    al, note = _border_matte(flat)
    if al is not None:
        notes.append("matte: border flood fill (busy background - install rembg or pass matte_file= for a cleaner cut)")
        if note:
            notes.append(note)
        return rgb, np.asarray(al, np.float32) / 255.0, "border", notes, None
    notes.append("matte: no foreground found - using the whole picture")
    return rgb, np.ones(rgb.shape[:2], np.float32), "none", notes, None


# ------------------------------------------------------------------------------------------------ analysis
def shape_metrics(alpha: np.ndarray) -> Dict[str, float]:
    from scipy.spatial import ConvexHull
    ndi = _ndi()
    hard = alpha >= 0.5
    ys, xs = np.nonzero(hard)
    if len(xs) == 0:
        raise ValueError("empty silhouette")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    area = float(hard.sum())
    edge = hard & ~ndi.binary_erosion(hard)
    ey, ex = np.nonzero(edge)
    pts = np.c_[ex, ey].astype(np.float64)
    try:
        hull = ConvexHull(pts)
        solidity = area / max(hull.volume, 1.0)
    except Exception:
        solidity = 1.0
    crop = hard[y0:y1, x0:x1]
    sym = float((crop & crop[:, ::-1]).sum() / max((crop | crop[:, ::-1]).sum(), 1))
    # thinness: the largest inscribed disc relative to the longest side (a scythe is thin, a dress is not)
    dt = ndi.distance_transform_edt(hard)
    thick = float(dt.max() * 2.0 / max(bw, bh))
    # share of the silhouette that lies on the central 20 % column band (long thin vertical things)
    return dict(x0=int(x0), x1=int(x1), y0=int(y0), y1=int(y1), width=int(bw), height=int(bh), area=area,
                fill=area / float(bw * bh), aspect=float(bh) / float(bw), solidity=float(solidity), symmetry=sym,
                thickness=thick)


def classify(metrics: Dict[str, float], matte: str, disc) -> Tuple[str, str]:
    m = metrics
    if disc is not None:
        return "sphere", f"round silhouette (ellipse axes {disc[2]:.0f} x {disc[3]:.0f} px)"
    if m["thickness"] < 0.07 or (m["fill"] < 0.22 and m["aspect"] > 1.6):
        return "prop", (f"thin and long (largest inscribed disc is {100 * m['thickness']:.1f} % of the length, "
                        f"fill {100 * m['fill']:.0f} %)")
    if m["aspect"] < 0.85 and m["symmetry"] > 0.8 and m["solidity"] > 0.9:
        return "arena", f"wide, left-right symmetric ring-like outline (w/h {1 / m['aspect']:.2f}, symmetry {m['symmetry']:.2f})"
    if matte in ("rembg", "border", "file") and m["solidity"] < 0.9 and m["aspect"] > 0.9:
        return "island", "cut out of a busy background with a wide, tapering outline (floating-island/landscape-like)"
    return "character", f"upright figure (h/w {m['aspect']:.2f}, symmetry {m['symmetry']:.2f})"


def disc_from_alpha(alpha: np.ndarray):
    """(cx, cy, rx, ry) of the axis-aligned ellipse that best matches a cut-out's outline (a sphere asked for by name on a
    picture that came with its own alpha / matte), or a bounding-box ellipse when the fit fails."""
    ndi = _ndi()
    m = alpha >= 0.5
    if not m.any():
        raise ValueError("the matte is empty - nothing to build a sphere from")
    edge = m & ~ndi.binary_erosion(m)
    ey, ex = np.nonzero(edge)
    fit = _fit_axis_ellipse(ex.astype(np.float64) + 0.5, ey.astype(np.float64) + 0.5) if len(ex) >= 40 else None
    if fit is None or not np.isfinite(fit).all() or min(fit[2], fit[3]) < 2:
        ys, xs = np.nonzero(m)
        return (0.5 * (xs.min() + xs.max() + 1), 0.5 * (ys.min() + ys.max() + 1),
                0.5 * (xs.max() - xs.min() + 1), 0.5 * (ys.max() - ys.min() + 1))
    return tuple(float(x) for x in fit)


def load_subject(src, matte: str = "auto", matte_file=None, kind: str = "auto", max_side: int = 2048,
                 name: Optional[str] = None, rembg_model: str = "u2net") -> Subject:
    img = load_rgba(src, max_side)
    rgb, alpha, method, notes, disc = make_alpha(img, matte, matte_file, rembg_model)
    met = shape_metrics(alpha)
    if kind == "auto":
        kind, why = classify(met, method, disc)
    else:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS} or 'auto'")
        why = "chosen by you"
    nm = name or (Path(src).stem if isinstance(src, (str, Path)) else "asset")
    if kind == "sphere" and disc is None:
        disc = disc_from_alpha(alpha)
    return Subject(rgb=rgb, alpha=alpha, matte=method, notes=notes, kind=kind, kind_reason=why, metrics=met,
                   disc=disc if kind == "sphere" else None, name=nm)
