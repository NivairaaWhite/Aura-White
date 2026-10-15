"""Outline extraction for *any* kind of picture.

The silhouette is the most valuable single fact about a picture - every later stage (3-D generation, view fitting,
carving, texture projection) leans on it, and a wrong one is the usual reason a character comes out as a blob with the
picture pasted on it.  So this module does not trust one method.  It

  1. looks at the picture (transparent / flat-colour illustration / line art / photograph) - only to decide the *order*
     in which methods are tried, never to decide the result;
  2. runs every method that applies: the alpha channel, the exact plain-background matte, a border flood fill, a
     line-art interior fill, learned segmenters (isnet-anime, birefnet, isnet-general, u2net through `rembg`, when
     installed), and a GrabCut refinement seeded by the best candidates (when OpenCV is installed);
  3. scores every candidate *without a ground truth* - does its boundary sit on real image edges, is it one solid piece,
     does it leak out of the frame, how much do the other methods agree - and adds a majority-vote candidate;
  4. returns the best, with a soft alpha (guided filter) and de-fringed colours - or raises `OutlineError`.

It never quietly returns the whole picture as the object: that is exactly the failure that produces a pasted slab.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except Exception:                                          # pragma: no cover
    ndi = None

BORDER_FLOOD_MIN_UNIFORM = 0.60
GOOD_ENOUGH = 0.80          # a candidate scoring at least this ends the (slower) learned-model search early
LONE_CONSENSUS = 0.40      # what a candidate with no independent second opinion gets
MIN_SCORE = 0.38            # below this nothing is trustworthy: raise instead of guessing
MODEL_ORDER = {
    "illustration": ("isnet-anime", "birefnet-general", "isnet-general-use", "u2net"),
    "lineart": ("isnet-anime", "isnet-general-use", "birefnet-general"),
    "photo": ("birefnet-general", "isnet-general-use", "u2net", "isnet-anime"),
    "transparent": (), "plain": ("isnet-general-use", "birefnet-general"),
}


class OutlineError(RuntimeError):
    pass


@dataclass
class Outline:
    alpha: np.ndarray                    # (H, W) float32 in [0, 1]
    hard: np.ndarray                     # (H, W) bool
    rgb: np.ndarray                      # (H, W, 3) uint8, de-fringed
    kind: str                            # transparent | plain | lineart | illustration | photo
    method: str                          # the winning candidate
    score: float                         # 0..1
    candidates: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def rgba(self) -> Image.Image:
        out = Image.fromarray(self.rgb, "RGB").convert("RGBA")
        out.putalpha(Image.fromarray((np.clip(self.alpha, 0, 1) * 255 + 0.5).astype(np.uint8)))
        return out


# ------------------------------------------------------------------------------------------------ classification
def classify_image(rgb: np.ndarray, has_alpha: bool = False) -> Tuple[str, Dict[str, float]]:
    """Heuristic picture type.  Used only to order the methods; every method is scored on the result anyway."""
    a = np.asarray(rgb[..., :3], np.uint8)
    h, w, _ = a.shape
    b = max(2, int(0.012 * min(h, w)))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)]).astype(np.float32)
    bg = np.median(border, 0)
    uniform = float(np.mean(np.linalg.norm(border - bg, axis=1) < 18.0))
    f = a.reshape(-1, 3).astype(np.int32)
    mx, mn = f.max(1), f.min(1)
    sat = float(np.mean((mx - mn) / np.maximum(mx, 1)))
    gray = f.mean(1)
    ink = float(np.mean(gray < 90))
    paper = float(np.mean(gray > 200))
    q = (a[::2, ::2] >> 3).reshape(-1, 3)                                      # 5 bits / channel
    colours = int(len(np.unique(q[:, 0].astype(np.int32) * 1024 + q[:, 1].astype(np.int32) * 32 + q[:, 2])))
    feats = {"border_uniform": uniform, "saturation": sat, "ink": ink, "paper": paper, "colours": float(colours)}
    if has_alpha:
        return "transparent", feats
    if sat < 0.10 and 0.002 < ink < 0.30 and (paper > 0.55 or ink > 0.55) and colours < 700:
        return "lineart", feats
    if uniform >= 0.90:
        return "plain", feats
    if colours < 2600:
        return "illustration", feats
    return "photo", feats


# ------------------------------------------------------------------------------------------------ strategies
def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _largest(mask: np.ndarray, keep_share: float = 0.002) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes >= max(1.0, keep_share * mask.sum())) + 1
    return np.isin(lab, keep)


def _alpha_mask(rgba: np.ndarray) -> Optional[np.ndarray]:
    a = rgba[..., 3]
    clear = float(np.mean(a < 250))
    if clear < 0.005 or clear > 0.995:
        return None
    return a >= 128


def _plain_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    from ..image.matte import plain_background_matte
    m = plain_background_matte(rgb)
    return None if m is None else (m.alpha >= 0.5)


def _border_flood_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Flood the background in from the border: pixels connected to it whose colour stays close to the colour they touch."""
    work = rgb.astype(np.float32)
    h, w, _ = work.shape
    b = max(2, int(0.012 * min(h, w)))
    border = np.concatenate([work[:b].reshape(-1, 3), work[-b:].reshape(-1, 3), work[:, :b].reshape(-1, 3), work[:, -b:].reshape(-1, 3)])
    bg = np.median(border, 0)
    spread = np.linalg.norm(border - bg, axis=1)
    thr = float(np.clip(np.percentile(spread, 92) * 1.5 + 10.0, 16.0, 80.0))
    dist = np.linalg.norm(work - bg, axis=2)
    cand = dist <= thr
    lab, _ = ndi.label(cand)
    edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bgm = np.isin(lab, edge_labels[edge_labels > 0])
    fg = ~bgm
    if fg.mean() < 0.004 or fg.mean() > 0.97:
        return None
    fg = ndi.binary_opening(fg, iterations=1)
    return _largest(ndi.binary_fill_holes(fg) if fg.mean() < 0.9 else fg)


def _lineart_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Line art: the strokes are the walls, the outside is whatever the border can reach, the figure is the rest."""
    g = _gray(rgb)
    light_bg = float(np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]))) > 127
    ink = (g < 0.55 * 255 if light_bg else g > 0.45 * 255)
    h, w = g.shape
    base_r = max(1, int(round(0.004 * min(h, w))))
    for r in (base_r, 2 * base_r, 3 * base_r, 5 * base_r, 8 * base_r):
        walls = ndi.binary_dilation(ink, structure=ndi.generate_binary_structure(2, 2), iterations=r)
        free = ~walls
        lab, _ = ndi.label(free)
        outside = np.isin(lab, np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])[:]))
        outside &= free
        fg = ~outside
        frac = float(fg.mean())
        if 0.01 < frac < 0.9:
            fg = ndi.binary_erosion(fg, structure=ndi.generate_binary_structure(2, 2), iterations=max(0, r - 1))
            fg = ndi.binary_fill_holes(fg | ink)
            return _largest(fg)
    return None


_SESSIONS: Dict[str, object] = {}


def _rembg_mask(rgb: np.ndarray, model: str) -> Optional[np.ndarray]:
    try:
        import rembg
    except Exception:
        return None
    try:
        sess = _SESSIONS.get(model)
        if sess is None:
            sess = _SESSIONS[model] = rembg.new_session(model)
        out = rembg.remove(Image.fromarray(rgb), session=sess, only_mask=True)
        m = np.asarray(out.convert("L") if hasattr(out, "convert") else out)
        return m >= 128
    except Exception:
        return None


def _grabcut_mask(rgb: np.ndarray, seed: Optional[np.ndarray]) -> Optional[np.ndarray]:
    try:
        import cv2
    except Exception:
        return None
    h, w = rgb.shape[:2]
    scale = 640.0 / max(h, w)
    small = cv2.resize(rgb, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    sh, sw = small.shape[:2]
    gc = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    if seed is None or not seed.any():
        gc[int(0.06 * sh):int(0.94 * sh), int(0.06 * sw):int(0.94 * sw)] = cv2.GC_PR_FGD
    else:
        s = np.asarray(Image.fromarray((seed * 255).astype(np.uint8)).resize((sw, sh), Image.NEAREST)) > 127
        r = max(2, int(0.03 * min(sh, sw)))
        outer = ndi.binary_dilation(s, iterations=r)
        inner = ndi.binary_erosion(s, iterations=r)
        gc[outer] = cv2.GC_PR_FGD
        gc[s] = cv2.GC_PR_FGD
        gc[inner] = cv2.GC_FGD
        gc[~outer] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return None
    m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    if sh != h:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) > 127
    return m


# ------------------------------------------------------------------------------------------------ scoring
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def score_mask(mask: np.ndarray, rgb: np.ndarray, others: Sequence[np.ndarray] = ()) -> Tuple[float, Dict[str, float]]:
    """0..1 plausibility of a silhouette, without a reference.  Higher is better."""
    m = np.asarray(mask, bool)
    area = float(m.mean())
    if area < 0.004 or area > 0.975:
        return 0.0, {"area": area}
    g = _gray(rgb)
    gm = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))
    gm = gm / max(float(np.percentile(gm, 97)), 1e-6)
    band = m & ~ndi.binary_erosion(m)
    band = ndi.binary_dilation(band)                                       # +-1 px: tolerant of 1 px offsets
    ratio = float(gm[band].mean() / max(gm.mean(), 1e-6)) if band.any() else 0.0
    edge = 1.0 - float(np.exp(-ratio / 2.2))
    lab, n = ndi.label(m)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([1])
    solid = float(sizes.max() / max(sizes.sum(), 1))
    b = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    leak = float(b.mean())
    border_ok = 1.0 - min(1.0, max(0.0, leak - 0.05) / 0.45)
    cons = float(np.median([_iou(m, o) for o in others])) if len(others) else LONE_CONSENSUS
    base = 0.45 * edge + 0.25 * solid + 0.30 * border_ok
    total = base * (0.45 + 0.55 * cons)                         # agreement of independent methods scales the whole score
    return float(total), {"edge": edge, "consensus": cons, "solid": solid, "border": border_ok, "area": area}


# ------------------------------------------------------------------------------------------------ refinement
def guided_alpha(mask: np.ndarray, rgb: np.ndarray, radius: Optional[int] = None, eps: float = 1e-3) -> np.ndarray:
    """Soft alpha: a guided filter (the picture steers the edge) applied in a thin band around the hard mask."""
    h, w = mask.shape
    r = radius or max(2, int(round(0.004 * min(h, w))))
    I = _gray(rgb) / 255.0
    p = mask.astype(np.float32)
    k = 2 * r + 1
    mean = lambda x: ndi.uniform_filter(x, size=k, mode="nearest")
    mI, mp = mean(I), mean(p)
    cov = mean(I * p) - mI * mp
    var = mean(I * I) - mI * mI
    a = cov / (var + eps)
    bb = mp - a * mI
    q = mean(a) * I + mean(bb)
    band = ndi.binary_dilation(mask, iterations=r) & ~ndi.binary_erosion(mask, iterations=r)
    out = np.where(band, np.clip((q - 0.5) * 1.6 + 0.5, 0.0, 1.0), p)
    out = np.where(mask & ~band, 1.0, out)
    out = np.where(~mask & ~band, 0.0, out)
    return out.astype(np.float32)


def defringe(rgb: np.ndarray, alpha: np.ndarray, thr: float = 0.98) -> np.ndarray:
    """Replace colours of semi-transparent edge pixels by the nearest fully opaque colour (removes background halos)."""
    solid = alpha >= thr
    if solid.all() or not solid.any():
        return rgb
    _, (iy, ix) = ndi.distance_transform_edt(~solid, return_indices=True)
    out = rgb.copy()
    edge = ~solid & (alpha > 0.02)
    out[edge] = rgb[iy[edge], ix[edge]]
    return out


def _clean(mask: np.ndarray, hole_frac: float = 0.004) -> np.ndarray:
    m = _largest(np.asarray(mask, bool))
    filled = ndi.binary_fill_holes(m)
    holes = filled & ~m
    if holes.any():
        lab, n = ndi.label(holes)
        sizes = np.bincount(lab.ravel())[1:]
        small = np.flatnonzero(sizes <= hole_frac * max(float(m.sum()), 1.0)) + 1
        m = m | np.isin(lab, small)
    return m


# ------------------------------------------------------------------------------------------------ the extractor
def extract_outline(image, kind: str = "auto", models: Optional[Sequence[str]] = None, strict: bool = True,
                    max_side: int = 1536, use_models: bool = True, say: Optional[Callable[[str], None]] = None) -> Outline:
    """Outline of the subject of `image` (PIL image or path).  `strict=True` raises `OutlineError` instead of ever
    returning the whole frame."""
    if ndi is None:
        raise OutlineError("SciPy is required for outline extraction (pip install scipy)")
    say = say or (lambda *_: None)
    img = Image.open(image) if not isinstance(image, Image.Image) else image
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    rgba = np.asarray(img.convert("RGBA"))
    rgb = rgba[..., :3].copy()
    has_alpha = img.mode in ("RGBA", "LA") and _alpha_mask(rgba) is not None
    detected, feats = classify_image(rgb, has_alpha)
    k = detected if kind == "auto" else kind
    notes = [f"picture looks like: {detected}" + ("" if kind == "auto" else f" (you said: {kind})")]
    cands: Dict[str, np.ndarray] = {}

    def add(name: str, m: Optional[np.ndarray]):
        if m is not None and m.shape == rgb.shape[:2] and m.any():
            cands[name] = _clean(m)

    if has_alpha:
        # The author's own alpha channel is ground truth, not one opinion among several.  (Competing it against the other
        # methods is a trap: under a transparent area the colour is often black, so "plain background" methods agree on a
        # wrong rectangle and outvote it.)
        hard = _clean(_alpha_mask(rgba))
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        notes.append("outline: the picture's own alpha channel")
        say(notes[-1])
        return Outline(alpha, hard, rgb, "transparent", "alpha", 1.0, {"alpha": 1.0}, notes)
    strategies = [("plain", _plain_mask)]
    if feats["border_uniform"] >= BORDER_FLOOD_MIN_UNIFORM:        # a flood fill is meaningless when the border is clutter
        strategies.append(("border", _border_flood_mask))
    for name, fn in strategies:
        try:
            add(name, fn(rgb))
        except Exception as e:                                 # a strategy failing must not stop the others
            notes.append(f"{name} matte failed: {type(e).__name__}")
    if k == "lineart" or detected == "lineart":
        try:
            add("lineart", _lineart_mask(rgb))
        except Exception as e:
            notes.append(f"line-art fill failed: {type(e).__name__}")

    def independent(name: str) -> bool:
        return not (name == "vote" or name.endswith("-seeded"))      # derived from other candidates: not a second opinion

    def ranked() -> List[Tuple[str, float]]:
        out = []
        for name, m in cands.items():
            others = [o for n2, o in cands.items() if n2 != name and independent(n2) and independent(name)]
            if not independent(name):                                # judge a derived candidate against the independent ones
                others = [o for n2, o in cands.items() if independent(n2)]
            out.append((name, score_mask(m, rgb, others)[0]))
        return sorted(out, key=lambda t: -t[1])

    top = ranked()
    if use_models and (not top or top[0][1] < GOOD_ENOUGH) and k != "transparent":
        for model in (models or MODEL_ORDER.get(k, MODEL_ORDER["photo"])):
            m = _rembg_mask(rgb, model)
            if m is None:
                continue
            add(f"rembg:{model}", m)
            top = ranked()
            if top and top[0][1] >= GOOD_ENOUGH:
                break
    if not has_alpha:
        add("grabcut", _grabcut_mask(rgb, None))                # independent: only a "the subject is roughly central" prior
        top = ranked()
        if top:
            add(f"{top[0][0]}-seeded", _grabcut_mask(rgb, cands[top[0][0]]))      # refines the best candidate's edge
    ind = [m for n, m in cands.items() if independent(n)]
    if len(ind) >= 3:                                           # majority vote of what exists
        add("vote", np.stack(ind).mean(0) >= 0.5)
    top = ranked()
    scores = {n: round(s, 3) for n, s in top}
    if not top or top[0][1] < MIN_SCORE:
        msg = ("could not find a trustworthy outline " + (f"(best candidate {top[0][0]} scored {top[0][1]:.2f})" if top else "(no method produced one)")
               + ". Provide a transparent PNG or a mask, or install the learned segmenters: pip install rembg onnxruntime")
        if strict:
            raise OutlineError(msg)
        notes.append(msg + " - using the whole picture")
        hard = np.ones(rgb.shape[:2], bool)
        return Outline(hard.astype(np.float32), hard, rgb, k, "none", 0.0, scores, notes)
    best, best_score = top[0]
    if not independent(best):
        # a refined / voted mask must beat the exact methods, not merely tie them: GrabCut can erode hairlines
        for exact in ("alpha", "plain", "lineart", "border"):
            if exact in cands and _iou(cands[exact], cands[best]) >= 0.97 and dict(top).get(exact, 0) >= MIN_SCORE:
                best, best_score = exact, dict(top)[exact]
                break
    hard = cands[best]
    alpha = rgba[..., 3].astype(np.float32) / 255.0 if best == "alpha" else guided_alpha(hard, rgb)
    notes.append(f"outline: {best} (score {best_score:.2f}; " + ", ".join(f"{n} {s:.2f}" for n, s in top[1:4]) + ")")
    say(notes[-1])
    return Outline(alpha, hard, defringe(rgb, alpha), k, best, float(best_score), scores, notes)
