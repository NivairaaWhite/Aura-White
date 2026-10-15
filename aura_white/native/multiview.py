"""Turn a character / prop SHEET (front, back, left, right pictures) into a real, closed, textured 3-D model - no neural
network, no GPU, no third-party weights.

Why this exists: a single picture cannot say what the hidden sides look like, so a model has to GUESS them (and guessing
is what produced "a picture pasted on a slab" and "the same image on both sides").  When the artist has already drawn the
other sides, nothing has to be guessed:

  1. matte     each picture -> an outline (soft, sub-pixel);  small specks (watermarks, logos) are dropped
  2. register  every picture is moved into ONE shared orthographic frame (same height, feet on the ground, the same
               centre line); front vs mirrored-back and left vs mirrored-right are used to check / refine the alignment
  3. carve     for every height, the front/back outlines say how wide the object is and the left/right outlines say how
               deep it is.  The cross-section is the round shape that touches both (a super-ellipse), not a box, so
               sleeves, skirts, torsos and hair come out rounded instead of "a stack of rectangles"
  4. mesh      smooth field -> marching cubes -> closed manifold mesh -> quadric decimation to the requested face budget
  5. unwrap    xatlas gives a clean, non-overlapping UV layout
  6. paint     every texel is coloured from the pictures that really see it: weight = how squarely the surface faces the
               camera, zero where the surface is hidden behind another part.  The back of the model comes from the BACK
               picture, the sides from the side pictures.
  7. check     the finished mesh is rendered back into every input view and compared with the picture.

What it can NOT do (said plainly): the body's depth between the views is an interpolation (a rounded cross-section), so
the shape is exactly right in the four outlines and a plausible rounding in between; fine relief that is not visible in an
outline (folds, a nose) comes from the pictures' paint, not from geometry; two pictures that disagree about the outline
(a model drew the back 3% narrower than the front) are reconciled by intersecting them.

numpy + scipy + opencv + scikit-image + xatlas + fast-simplification.  Nothing here imports torch.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..gen3d.views import ViewCam, VIEW_ANGLES, canonical_label, infer_label, render_maps

REQUIRED_ANY = ("front", "back")                     # at least one view says how wide
SIDES = ("left", "right")                            # at least one view says how deep
FILL = 0.9                                           # the object's height is this share of the canvas
SAY = Callable[[str], None]


# ------------------------------------------------------------------------------------------------ pictures -> mattes
def load_image(src: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """-> (rgb float32 (H,W,3) in [0,1], alpha float32 (H,W) or None when the picture has no transparency)."""
    from PIL import Image

    if isinstance(src, np.ndarray):
        a = src.astype(np.float32)
        a = a / 255.0 if a.max() > 1.5 else a
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, 2)
        if a.shape[-1] == 4:
            return a[..., :3].copy(), a[..., 3].copy()
        return a[..., :3].copy(), None
    im = Image.open(str(src))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        a = np.asarray(im).astype(np.float32) / 255.0
        alpha = a[..., 3]
        if alpha.min() < 0.99:
            return a[..., :3].copy(), alpha.copy()
        return a[..., :3].copy(), None
    return np.asarray(im.convert("RGB")).astype(np.float32) / 255.0, None


def extract_matte(rgb: np.ndarray, alpha: Optional[np.ndarray] = None, min_share: float = 0.02,
                  lo: float = 0.06, hi: float = 0.16) -> np.ndarray:
    """Soft matte (H,W) float32 in [0,1].  Uses the alpha channel when there is one, else the distance from the border
    colour.  Holes are filled and components smaller than `min_share` of the biggest are dropped (watermarks, logos)."""
    from scipy import ndimage as ndi

    if alpha is not None:
        soft = np.clip(alpha, 0.0, 1.0)
    else:
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bg = np.median(border, 0)
        d = np.abs(rgb - bg).max(-1)
        soft = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    hard = soft > 0.5
    hard = ndi.binary_fill_holes(hard)
    lab, n = ndi.label(hard)
    if n == 0:
        return soft.astype(np.float32)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes >= min_share * sizes.max()) + 1
    kept = np.isin(lab, keep)
    near = ndi.binary_dilation(kept, iterations=3)
    out = np.where(kept, 1.0, np.where(near, soft, 0.0))              # holes -> 1, rim keeps its soft edge
    return out.astype(np.float32)


def _bbox_rows(m: np.ndarray, thr: float = 0.5, min_px: int = 2) -> Tuple[int, int]:
    rows = np.flatnonzero((m > thr).sum(1) >= min_px)
    return int(rows[0]), int(rows[-1] + 1)


def _foot_centre(m: np.ndarray, top: int, bottom: int, share: float = 0.025) -> float:
    """x of the middle of the feet: the bottom `share` of the object's height."""
    h = bottom - top
    r0 = max(top, int(bottom - max(3, share * h)))
    cols = np.flatnonzero((m[r0:bottom] > 0.5).any(0))
    if len(cols) == 0:
        cols = np.flatnonzero((m > 0.5).any(0))
    return float((cols[0] + cols[-1] + 1) / 2.0)


def _fill_outside(rgb: np.ndarray, matte: np.ndarray, erode: int = 2) -> np.ndarray:
    """Replace everything that is not safely inside the object by the nearest inside colour (no dark halo, no black bleed)."""
    from scipy import ndimage as ndi

    inside = ndi.binary_erosion(matte > 0.5, iterations=erode)
    if not inside.any():
        return rgb
    _, idx = ndi.distance_transform_edt(~inside, return_indices=True)
    return rgb[idx[0], idx[1]]


# ------------------------------------------------------------------------------------------------ registration
@dataclass
class RegView:
    label: str
    az: float
    rgb: np.ndarray             # (S,S,3) float32, outside filled with the nearest object colour
    matte: np.ndarray           # (S,S) float32 soft outline
    cam: ViewCam                # orthographic camera shared by all views (only az differs)
    src_size: Tuple[int, int]
    info: Dict = field(default_factory=dict)


def canvas_cam(az: float, S: int) -> ViewCam:
    """World frame: y in [0,1] (feet at 0), x = 0 / z = 0 on the centre line.  scale = FILL*S pixels per unit of height."""
    return ViewCam(az=az, el=0.0, size=S, scale=FILL * S, ox=0.0, oy=0.5)


def _warp(img: np.ndarray, s: float, tx: float, ty: float, S: int, border=0.0) -> np.ndarray:
    """Scale by s and move (tx, ty).  warpAffine has no area filter, so a shrinking warp is pre-blurred."""
    import cv2

    if s < 1.0:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.45 / s)
    M = np.array([[s, 0.0, tx - 0.5 * (1 - s) * 0.0], [0.0, s, ty]], np.float64)
    # pixel-centre convention: cv2 maps integer coordinates to pixel centres
    M[0, 2] = tx + 0.5 * s - 0.5
    M[1, 2] = ty + 0.5 * s - 0.5
    return cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _mirror_search(ref: np.ndarray, mov: np.ndarray, max_dx: int, max_dy: int, scale: int = 4) -> Tuple[int, int, float, float]:
    """Find the shift (dx, dy) of `mov` (already mirrored to look like `ref`) that maximises the outline overlap.
    -> (dx, dy, iou_before, iou_after) in canvas pixels."""
    import cv2

    a = cv2.resize((ref > 0.5).astype(np.float32), None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_AREA) > 0.5
    b = cv2.resize((mov > 0.5).astype(np.float32), None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_AREA) > 0.5

    def iou(dx, dy):
        bb = np.roll(np.roll(b, dy, 0), dx, 1)
        u = (a | bb).sum()
        return float((a & bb).sum() / u) if u else 0.0

    base = iou(0, 0)
    best = (0, 0, base)
    mx, my = max(1, max_dx // scale), max(1, max_dy // scale)
    for dy in range(-my, my + 1):
        for dx in range(-mx, mx + 1):
            v = iou(dx, dy)
            if v > best[2] + 1e-6:
                best = (dx, dy, v)
    return best[0] * scale, best[1] * scale, base, best[2]


def prepare_views(sources: Dict[str, Union[str, Path, np.ndarray]], S: int = 1536, say: Optional[SAY] = None,
                  mattes: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, RegView]:
    """Load, matte and register the pictures.  `sources` maps a view name ('front', 'back', 'left', 'right', or
    'name@azimuth') to a path or array.  Only the four orthographic side views take part in the carving; any other
    angle is kept for painting only."""
    say = say or (lambda *_: None)
    raw: Dict[str, Dict] = {}
    for name, src in sources.items():
        lab = canonical_label(name.partition("@")[0])
        if lab is None:
            raise ValueError(f"unknown view {name!r}: use front, back, left, right")
        rgb, alpha = load_image(src)
        m = (mattes or {}).get(name)
        matte = extract_matte(rgb, alpha) if m is None else np.asarray(m, np.float32)
        if float((matte > 0.5).mean()) < 0.002:
            raise ValueError(f"no subject found in the {name!r} picture (nothing stands out from the background); "
                             "use a picture of the character on a plain or transparent background")
        top, bot = _bbox_rows(matte)
        raw[lab] = {"rgb": rgb, "matte": matte, "top": top, "bot": bot, "cx": _foot_centre(matte, top, bot)}
    out: Dict[str, RegView] = {}
    tf: Dict[str, Tuple[float, float, float]] = {}
    for lab, r in raw.items():
        s = FILL * S / max(r["bot"] - r["top"], 1)
        tx = S / 2.0 - s * r["cx"]
        ty = 0.95 * S - s * r["bot"]
        tf[lab] = (s, tx, ty)
    # mirror refinement: back against front, right against left  (their outlines are mirror images of each other)
    pairs = [("back", "front"), ("right", "left")]
    info: Dict[str, Dict] = {lab: {} for lab in raw}
    for mov, ref in pairs:
        if mov in raw and ref in raw:
            mr = _warp(raw[ref]["matte"], *tf[ref], S)
            mm = _warp(raw[mov]["matte"], *tf[mov], S)[:, ::-1]
            dx, dy, before, after = _mirror_search(mr, mm, max_dx=int(0.06 * S), max_dy=int(0.015 * S))
            # the mirrored copy moved by dx  <=>  the real one moved by -dx
            s, tx, ty = tf[mov]
            tf[mov] = (s, tx - dx, ty + dy)
            info[mov].update({"mirror_iou_before": round(before, 4), "mirror_iou_after": round(after, 4),
                              "shift_px": (int(-dx), int(dy))})
            say(f"  {mov} vs mirrored {ref}: outline overlap {before:.3f} -> {after:.3f} (moved {-dx},{dy}px)")
    for lab, r in raw.items():
        s, tx, ty = tf[lab]
        matte = np.clip(_warp(r["matte"], s, tx, ty, S), 0, 1).astype(np.float32)
        rgb = _warp(r["rgb"], s, tx, ty, S, border=0.0).astype(np.float32)
        rgb = _fill_outside(rgb, matte)
        az = VIEW_ANGLES[lab][0]
        info[lab].update({"scale": round(s, 4), "src": list(r["rgb"].shape[:2][::-1])})
        out[lab] = RegView(lab, az, rgb, matte, canvas_cam(az, S), tuple(r["rgb"].shape[:2][::-1]), info[lab])
    return out


# ------------------------------------------------------------------------------------------------ the carve
def _profile(view: RegView, xs: np.ndarray, ys: np.ndarray, flip: bool) -> np.ndarray:
    """Sample the outline of `view` on a (len(ys), len(xs)) grid of world coordinates; `xs` are the coordinates of the
    horizontal axis of that view (x for front/back, z for left/right)."""
    import cv2

    c = view.cam
    xr = -xs if flip else xs
    px = (c.size / 2.0 + c.scale * (xr - c.ox) - 0.5).astype(np.float32)
    py = (c.size / 2.0 - c.scale * (ys - c.oy) - 0.5).astype(np.float32)
    mx, my = np.meshgrid(px, py)
    return cv2.remap(view.matte.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def _combine(profiles: List[np.ndarray]) -> np.ndarray:
    return profiles[0] if len(profiles) == 1 else np.minimum.reduce(profiles)


def _row_runs(row: np.ndarray, xs: np.ndarray, h: float, min_run: float) -> Tuple[np.ndarray, np.ndarray]:
    """The runs of one outline row: (centres, half-widths), edges interpolated to sub-voxel accuracy."""
    inside = row >= 0.5
    n = len(row)
    cs, hws = [], []
    if inside.any():
        d = np.diff(inside.astype(np.int8), prepend=0, append=0)
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):      # e is exclusive
            left = xs[s]
            if s > 0:
                a, b = row[s - 1], row[s]
                left = xs[s - 1] + (0.5 - a) / max(b - a, 1e-6) * h
            right = xs[e - 1]
            if e < n:
                a, b = row[e - 1], row[e]
                right = xs[e - 1] + (a - 0.5) / max(a - b, 1e-6) * h
            if right - left < min_run * h:
                continue
            cs.append(0.5 * (left + right))
            hws.append(0.5 * (right - left))
    return np.array(cs, np.float64), np.array(hws, np.float64)


def _nearest_run(xs: np.ndarray, cs: np.ndarray, hws: np.ndarray) -> np.ndarray:
    dist = np.maximum(np.abs(xs[:, None] - cs[None, :]) - hws[None, :], 0.0)
    return dist.argmin(1)


def _synth(G: np.ndarray, coords_from: np.ndarray, coords_to: np.ndarray, ratio: float) -> np.ndarray:
    """A stand-in outline for a view that was not supplied: per row, a band centred on 0 whose half-size is `ratio` times
    the half-size of the row of `G`.  Only used when a side or front view is missing - and reported as a guess."""
    out = np.zeros((G.shape[0], len(coords_to)), np.float32)
    for k in range(G.shape[0]):
        on = coords_from[G[k] >= 0.5]
        if len(on):
            half = ratio * 0.5 * (on.max() - on.min())
            out[k] = (np.abs(coords_to) <= half).astype(np.float32)
    return out


@dataclass
class Field:
    f: np.ndarray                 # (Ny, Nx, Nz) float32, < 0 inside
    h: float
    y0: float
    x0: float
    z0: float
    notes: List[str] = field(default_factory=list)


def build_field(views: Dict[str, RegView], cells: int = 420, roundness: float = 2.4, smooth: float = 1.0,
                min_run: float = 1.5, margin: int = 6, depth_ratio: float = 0.5, dominance: float = 0.6,
                limb_aspect: float = 1.4, wing_p: Optional[float] = None) -> Field:
    """Cross-sections from the outlines (see the module docstring).  `cells` = voxels along the object's height."""
    from scipy import ndimage as ndi

    notes: List[str] = []
    h = 1.0 / cells
    ys = (np.arange(-margin, cells + margin) + 0.5) * h
    wide = [v for k, v in views.items() if k in REQUIRED_ANY]
    deep = [v for k, v in views.items() if k in SIDES]
    if not wide and not deep:
        raise ValueError("need at least one of front/back (width) or left/right (depth)")

    def extent(vs: List[RegView], flips: List[bool]) -> Tuple[float, float]:
        lo, hi = 9.0, -9.0
        for v, fl in zip(vs, flips):
            cols = np.flatnonzero((v.matte > 0.5).any(0))
            if len(cols) == 0:
                continue
            xr = (cols - v.cam.size / 2.0) / v.cam.scale
            x = -xr if fl else xr
            lo, hi = min(lo, float(x.min())), max(hi, float(x.max()))
        return lo, hi

    if wide:
        flips = [v.label == "back" for v in wide]
        xlo, xhi = extent(wide, flips)
    else:
        xlo = xhi = 0.0
    if deep:
        flips = [v.label == "left" for v in deep]                    # z = -xr for the left view, +xr for the right
        zlo, zhi = extent(deep, flips)
    else:
        zlo = zhi = 0.0
    pad = margin * h

    def axis(lo: float, hi: float) -> np.ndarray:
        return np.arange(lo - pad, hi + pad, h) + 0.5 * h

    if not deep:
        notes.append("NO side view: the depth is a guess (a fixed share of the width) - this is NOT a real 3-D reconstruction")
        zhalf = depth_ratio * 0.5 * (xhi - xlo)
        zlo, zhi = -zhalf, zhalf
    if not wide:
        notes.append("NO front/back view: the width is a guess (a fixed share of the depth) - this is NOT a real 3-D reconstruction")
        xhalf = 0.5 * (zhi - zlo) / max(depth_ratio, 0.1)
        xlo, xhi = -xhalf, xhalf
    xs, zs = axis(xlo, xhi), axis(zlo, zhi)
    Gx = _combine([_profile(v, xs, ys, flip=(v.label == "back")) for v in wide]) if wide else None
    Gz = _combine([_profile(v, zs, ys, flip=(v.label == "left")) for v in deep]) if deep else None
    if Gz is None:                                               # symmetric guess: half-depth = ratio * half-width of the row
        Gz = _synth(Gx, xs, zs, depth_ratio)
    if Gx is None:
        Gx = _synth(Gz, zs, xs, 1.0 / max(depth_ratio, 0.1))
    if len(wide) == 1:
        notes.append(f"only the {wide[0].label} view gives the width (supply the other one to cross-check the outline)")
    if len(deep) == 1:
        notes.append(f"only the {deep[0].label} view gives the depth (supply the other one to cross-check the outline)")

    Ny, Nx, Nz = len(ys), len(xs), len(zs)
    F = np.full((Ny, Nx, Nz), 9.0, np.float32)
    p = float(roundness)
    n_limited = 0
    for k in range(Ny):
        cx, hx = _row_runs(Gx[k], xs, h, min_run)
        cz, hz = _row_runs(Gz[k], zs, h, min_run)
        if len(cx) == 0 or len(cz) == 0:
            continue
        ix = _nearest_run(xs, cx, hx)
        iz = _nearest_run(zs, cz, hz)
        u = (xs - cx[ix]) / np.maximum(hx[ix], 0.5 * h)                       # position inside the x-run
        # depth available to each (x-run, z-run) pair.  The side outline's run is the union of everything at this height;
        # a narrow x-run beside a much wider one (an arm next to a torso, a tassel) is a thin part, not a slab as deep
        # as the whole body: its depth is limited to `limb_aspect` times its own half-width.
        H = np.tile(hz[None, :], (len(cx), 1))
        thin = hx < dominance * hx.max()
        deep_z = hz >= dominance * hz.max()
        for i in np.flatnonzero(thin):
            H[i, deep_z] = np.minimum(H[i, deep_z], limb_aspect * hx[i])
        n_limited += int(thin.sum() > 0 and (H < hz[None, :] - 1e-9).any())
        Hm = np.maximum(H[ix][:, iz], 0.5 * h)                                  # (Nx, Nz)
        w = (zs - cz[iz])[None, :] / Hm
        if wing_p is None:
            a_ = (np.abs(u).clip(max=9.0) ** p).astype(np.float32)
            b_ = (np.abs(w).clip(max=9.0) ** p).astype(np.float32)
            F[k] = (a_[:, None] + b_) ** (1.0 / p) - 1.0
        else:
            # very wide, shallow sections (a torso with sleeves hanging beside it) are better described by a pinched
            # star than by a fat ellipse: the exponent falls from `roundness` to `wing_p` as width / depth goes 1.5 -> 2.6
            asp = (hx[ix][:, None] / np.maximum(H[ix][:, iz], 0.5 * h))
            t_ = np.clip((asp - 1.5) / 1.1, 0.0, 1.0)
            P = (p + (wing_p - p) * (t_ * t_ * (3 - 2 * t_))).astype(np.float32)
            a_ = np.abs(u).clip(max=9.0)[:, None].astype(np.float32)
            b_ = np.abs(w).clip(max=9.0).astype(np.float32)
            F[k] = (a_ ** P + b_ ** P) ** (1.0 / P) - 1.0
    if n_limited:
        notes.append(f"{n_limited} heights had a thin part beside a wide one (arm / tassel): its depth was limited to "
                     f"{limb_aspect:g}x its width instead of the whole body's depth")
    if smooth > 0:
        F = ndi.gaussian_filter(F, sigma=(1.5 * smooth, smooth, smooth), mode="nearest")
    return Field(F, h, float(ys[0]), float(xs[0]), float(zs[0]), notes)


def field_to_mesh(fld: Field) -> Tuple[np.ndarray, np.ndarray]:
    """Marching cubes on the zero level, outward normals, world frame (x, y, z)."""
    from skimage import measure

    f = np.pad(fld.f, 1, mode="constant", constant_values=1.0)
    v, faces, _, _ = measure.marching_cubes(f, level=0.0, spacing=(fld.h, fld.h, fld.h))
    # array axes are (y, x, z); -1 undoes the pad
    y = v[:, 0] + fld.y0 - fld.h              # y0/x0/z0 = coordinates of the first sample; the pad shifted everything by one cell
    x = v[:, 1] + fld.x0 - fld.h
    z = v[:, 2] + fld.z0 - fld.h
    verts = np.stack([x, y, z], 1).astype(np.float64)
    faces = faces.astype(np.int64)
    if _signed_volume(verts, faces) < 0:
        faces = faces[:, ::-1]
    return verts, faces


def _signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def clean_mesh(v: np.ndarray, f: np.ndarray, min_share: float = 0.002) -> Tuple[np.ndarray, np.ndarray]:
    """Drop degenerate faces and tiny floating pieces; keep the orientation outward."""
    from ..gen3d.meshutil import component_labels

    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = f[area > 1e-14]
    fl = component_labels(f, len(v))
    ids, counts = np.unique(fl, return_counts=True)
    keep = np.isin(fl, ids[counts >= min_share * len(f)])
    f = f[keep]
    used = np.unique(f)
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


def decimate(v: np.ndarray, f: np.ndarray, target_faces: int) -> Tuple[np.ndarray, np.ndarray]:
    if target_faces <= 0 or len(f) <= target_faces:
        return v, f
    import fast_simplification as fs

    nv, nf = fs.simplify(np.ascontiguousarray(v, np.float32), np.ascontiguousarray(f, np.int32),
                         target_reduction=float(1.0 - target_faces / len(f)), agg=7)
    return nv.astype(np.float64), nf.astype(np.int64)


# ------------------------------------------------------------------------------------------------ the paint
def _bilinear(img: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Bilinear lookup of arbitrarily many points; (px, py) use the package convention (pixel centres at +0.5).
    cv2.remap only takes maps smaller than 32767 per side, so the points are laid out as a wide rectangle."""
    import cv2

    n = len(px)
    W = 2048
    R = (n + W - 1) // W
    mx = np.zeros(R * W, np.float32)
    my = np.zeros(R * W, np.float32)
    mx[:n] = px - 0.5
    my[:n] = py - 0.5
    out = cv2.remap(img, mx.reshape(R, W), my.reshape(R, W), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(R * W, *img.shape[2:])[:n]


def _unwrap(v: np.ndarray, f: np.ndarray, size: int, padding: int = 6):
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(np.ascontiguousarray(v, np.float32), np.ascontiguousarray(f, np.uint32))
    co = xatlas.ChartOptions()
    po = xatlas.PackOptions()
    po.resolution = int(size)
    po.padding = int(padding)
    po.bilinear = True
    po.create_image = False
    atlas.generate(co, po, verbose=False)
    vmap, idx, uv = atlas[0]
    return vmap.astype(np.int64), idx.astype(np.int64), uv.astype(np.float64)


def bake(v: np.ndarray, f: np.ndarray, views: Dict[str, RegView], tex: int = 2048, sharp: float = 6.0,
         vis_eps: float = 0.012, vis_size: int = 1024, say: Optional[SAY] = None):
    """-> (verts2, faces2, uv, normals2, albedo uint8 (tex,tex,3), report).  Texels are painted from the views that see them."""
    from scipy import ndimage as ndi
    from ..asset.camera import rasterize_ortho
    from ..mesh.cleanup import vertex_normals

    say = say or (lambda *_: None)
    t0 = time.time()
    vn_src = vertex_normals(v, f)
    vmap, f2, uv = _unwrap(v, f, tex)
    v2 = v[vmap]
    n2 = vn_src[vmap]
    say(f"  unwrapped: {len(v)} -> {len(v2)} vertices, {time.time() - t0:.1f}s")
    # rasterise the UV layout: which triangle and which barycentric coords own each texel
    xy = np.stack([uv[:, 0] * tex, (1.0 - uv[:, 1]) * tex], 1)
    tri, bary, _ = rasterize_ortho(xy, np.zeros(len(xy)), f2, tex, tex)
    cov = tri >= 0
    ti = tri[cov]
    bc = bary[cov].astype(np.float64)
    fv = f2[ti]
    P = (v2[fv] * bc[..., None]).sum(1)                                      # texel position in the object frame
    N = (n2[fv] * bc[..., None]).sum(1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    acc = np.zeros((len(P), 3), np.float64)
    wsum = np.zeros(len(P), np.float64)
    seen_by: Dict[str, float] = {}
    best_w = np.zeros(len(P))
    best_rgb = np.zeros((len(P), 3))
    for lab, rv in views.items():
        cam = rv.cam
        a = np.radians(rv.az)
        d = np.array([np.sin(a), 0.0, np.cos(a)])
        cosn = N @ d
        # visibility: depth buffer of the mesh from this camera
        vc = ViewCam(rv.az, 0.0, vis_size, FILL * vis_size, 0.0, 0.5)
        _, dep, _, _ = render_maps(v2, f2, vc)
        px, py, t = vc.project(P)
        ix = np.clip(np.floor(px).astype(np.int64), 0, vis_size - 1)
        iy = np.clip(np.floor(py).astype(np.int64), 0, vis_size - 1)
        dmax = ndi.maximum_filter(np.nan_to_num(dep, nan=-1e9), size=3)[iy, ix]    # nearest surface in a 3x3 window
        vis = (dmax > -1e8) & (t >= dmax - vis_eps)
        # inside the picture's own outline, away from the soft rim
        cx, cy, _ = cam.project(P)
        inside = _bilinear(_erode(rv.matte, 3), cx, cy) > 0.99
        w = np.clip(cosn, 0.0, None) ** sharp * vis * inside
        col = _bilinear(rv.rgb, cx, cy).astype(np.float64)
        acc += col * w[:, None]
        wsum += w
        upd = w > best_w
        best_w[upd] = w[upd]
        best_rgb[upd] = col[upd]
        seen_by[lab] = float((w > 0.05).mean())
    ok = wsum > 1e-4
    rgb = np.zeros((len(P), 3))
    rgb[ok] = acc[ok] / wsum[ok, None]
    albedo = np.zeros((tex, tex, 3), np.float32)
    filled = np.zeros((tex, tex), bool)
    cov_idx = np.flatnonzero(cov.ravel())
    albedo.reshape(-1, 3)[cov_idx[ok]] = rgb[ok]
    filled.reshape(-1)[cov_idx[ok]] = True
    hidden = float((~ok).mean())
    # texels no picture sees (under the arms, the crown, inside the hem) take the nearest painted texel
    if (~filled & cov).any():
        _, idx = ndi.distance_transform_edt(~filled, return_indices=True)
        albedo = np.where((cov & ~filled)[..., None], albedo[idx[0], idx[1]], albedo)
        filled = filled | cov
    # gutters: spread painted texels outwards so bilinear filtering and mip-maps never pull in the background
    _, idx = ndi.distance_transform_edt(~filled, return_indices=True)
    near = ndi.distance_transform_edt(~filled) <= 8
    albedo = np.where((~filled & near)[..., None], albedo[idx[0], idx[1]], albedo)
    img = (np.clip(albedo, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    rep = {"texels_covered": int(cov.sum()), "unseen_share": round(hidden, 4),
           "seen_share_by_view": {k: round(x, 3) for k, x in seen_by.items()}, "seconds": round(time.time() - t0, 1)}
    uv_gltf = np.stack([uv[:, 0], 1.0 - uv[:, 1]], 1).astype(np.float32)          # glTF: origin top-left, v downwards
    return v2, f2, uv_gltf, n2, img, rep


def _erode(m: np.ndarray, px: int) -> np.ndarray:
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode((m > 0.5).astype(np.float32), k)


# ------------------------------------------------------------------------------------------------ checks and previews
def view_agreement(v: np.ndarray, f: np.ndarray, views: Dict[str, RegView], size: int = 768) -> Dict[str, float]:
    """Silhouette IoU between the mesh and each input outline, in the shared frame."""
    import cv2

    out = {}
    for lab, rv in views.items():
        cam = ViewCam(rv.az, 0.0, size, FILL * size, 0.0, 0.5)
        mask = render_maps(v, f, cam)[0]
        tgt = cv2.resize(rv.matte, (size, size), interpolation=cv2.INTER_AREA) > 0.5
        u = (mask | tgt).sum()
        out[lab] = float((mask & tgt).sum() / u) if u else 1.0
    return out


def render_textured(v: np.ndarray, f: np.ndarray, uv: np.ndarray, albedo: np.ndarray, az: float, el: float = 0.0,
                    size: int = 640, bg: float = 0.18, light: Tuple[float, float, float] = (0.4, 0.55, 0.75),
                    shade: bool = True) -> np.ndarray:
    """Quick textured preview (uint8 RGB): z-buffer + lambert."""
    from ..mesh.cleanup import vertex_normals

    vv = v[:, [0, 1, 2]]
    ext_y = vv[:, 1].max() - vv[:, 1].min()
    cam = ViewCam(az, el, size, 0.9 * size / max(ext_y, 1e-9), 0.0, float((vv[:, 1].max() + vv[:, 1].min()) / 2.0))
    mask, dep, tri, bary = render_maps(v, f, cam)
    img = np.full((size, size, 3), bg, np.float32)
    if not mask.any():
        return (img * 255).astype(np.uint8)
    t = tri[mask]
    b = bary[mask].astype(np.float64)
    fv = f[t]
    uvv = (uv[fv] * b[..., None]).sum(1)
    tex = albedo.shape[0]
    col = _bilinear(albedo.astype(np.float32) / 255.0, uvv[:, 0] * tex, uvv[:, 1] * tex).reshape(-1, 3)
    vn = vertex_normals(v, f).astype(np.float64)
    n = (vn[fv] * b[..., None]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    a, e = np.radians(az), np.radians(el)
    cdir = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    right = np.array([np.cos(a), 0.0, -np.sin(a)])
    up = np.cross(cdir, right)
    L = light[0] * (-right) + light[1] * up + light[2] * cdir
    L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0.0, 1.0)
    k = (0.45 + 0.55 * lam) if shade else np.ones_like(lam)
    img[mask] = np.clip(col * k[:, None], 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def turntable(v, f, uv, albedo, azs: Sequence[float] = (0, 45, 90, 135, 180, 225, 270, 315), size: int = 512) -> np.ndarray:
    tiles = [render_textured(v, f, uv, albedo, a, 0.0, size) for a in azs]
    cols = min(len(tiles), 4)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * size, cols * size, 3), 46, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = t
    return sheet


# ------------------------------------------------------------------------------------------------ one call
@dataclass
class MVResult:
    verts: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    normals: np.ndarray
    albedo: np.ndarray
    report: Dict
    glb: Optional[Path] = None


def load_mesh(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Any mesh file trimesh can read (glb / obj / ply / stl) -> (verts, faces); scenes are merged."""
    import trimesh

    m = trimesh.load(str(path), force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def register_mesh(v: np.ndarray, f: np.ndarray, views: Dict[str, RegView], say: Optional[SAY] = None,
                  size: int = 256) -> Tuple[np.ndarray, Dict]:
    """Move a mesh made by ANY generator into the sheet's frame (y up, feet at 0, unit height, front = +z).
    Tries the four quarter-turns about the vertical axis, keeps the one whose outlines agree best with the pictures, then
    nudges it sideways / front-back to maximise the agreement.  Reports what it did - a low final agreement means the
    mesh and the pictures do not show the same pose / proportions and the painting will be wrong."""
    from ..gen3d.views import rot_y

    say = say or (lambda *_: None)
    v = np.asarray(v, np.float64)
    v = v - [0.5 * (v[:, 0].max() + v[:, 0].min()), v[:, 1].min(), 0.5 * (v[:, 2].max() + v[:, 2].min())]
    v = v / max(v[:, 1].max(), 1e-12)
    from ..gen3d.views import light_mesh

    lv0, lf0 = light_mesh(v, f, 20_000)

    def score(vv: np.ndarray) -> float:
        a = view_agreement(vv, lf0, views, size)
        return float(np.mean(list(a.values())))

    best = (-1.0, 0.0, None)
    for deg in (0.0, 90.0, 180.0, 270.0):
        r = rot_y(deg)
        vv = lv0 @ r.T
        vv = vv - [0.5 * (vv[:, 0].max() + vv[:, 0].min()), vv[:, 1].min(), 0.5 * (vv[:, 2].max() + vv[:, 2].min())]
        sc = score(vv)
        if sc > best[0]:
            best = (sc, deg, vv)
    sc0, deg, vv = best
    say(f"  mesh turned {deg:g} deg about the vertical axis: outline agreement {sc0:.3f}")
    off = np.zeros(2)                                          # (dx, dz)
    step = 0.04
    cur = sc0
    for _ in range(6):
        improved = True
        while improved:
            improved = False
            for k in (0, 1):
                for sgn in (1.0, -1.0):
                    d = off.copy()
                    d[k] += sgn * step
                    cand = vv + [d[0], 0.0, d[1]]
                    c = score(cand)
                    if c > cur + 1e-4:
                        cur, off, improved = c, d, True
        step *= 0.5
    say(f"  nudged by ({off[0]:+.3f}, {off[1]:+.3f}): outline agreement {cur:.3f}")
    r = rot_y(deg)
    out = np.asarray(v, np.float64) @ r.T
    out = out - [0.5 * (out[:, 0].max() + out[:, 0].min()), out[:, 1].min(), 0.5 * (out[:, 2].max() + out[:, 2].min())]
    out = out + [off[0], 0.0, off[1]]
    return out, {"turn_deg": deg, "shift": [float(off[0]), float(off[1])], "agreement_before": round(sc0, 4),
                 "agreement": round(cur, 4)}


def reconstruct(sources: Dict[str, Union[str, Path, np.ndarray]], out_dir: Optional[Union[str, Path]] = None,
                name: str = "model", height_m: float = 1.0, faces: int = 60_000, cells: int = 420, roundness: float = 2.4,
                tex: int = 2048, canvas: int = 1536, unlit: bool = False, wing_p: Optional[float] = None,
                base_mesh: Optional[Union[str, Path, Tuple[np.ndarray, np.ndarray]]] = None,
                say: Optional[SAY] = None) -> MVResult:
    """sources -> textured GLB (+ preview sheet + JSON report) in `out_dir`.

    Without `base_mesh` the shape is carved from the outlines (see the module docstring).  With one (a mesh from TRELLIS,
    Hunyuan, TripoSR, Meshy, a sculpt ... anything) the shape is KEPT and only registered to the pictures and painted from
    them - so its back comes from the back picture and its sides from the side pictures."""
    say = say or (lambda *_: None)
    t0 = time.time()
    say("matting + registering the pictures")
    views = prepare_views(sources, canvas, say)
    notes: List[str] = []
    reg: Dict = {}
    if base_mesh is None:
        say(f"carving ({cells} cells high, roundness {roundness})")
        fld = build_field(views, cells=cells, roundness=roundness, wing_p=wing_p)
        v, f = field_to_mesh(fld)
        notes += fld.notes
    else:
        v, f = load_mesh(base_mesh) if isinstance(base_mesh, (str, Path)) else (np.asarray(base_mesh[0], np.float64), np.asarray(base_mesh[1], np.int64))
        say("registering the supplied mesh to the pictures")
        v, reg = register_mesh(v, f, views, say)
        if reg["agreement"] < 0.85:
            notes.append(f"the mesh and the pictures agree only {reg['agreement']:.0%} on the outline - they probably do not show "
                         "the same pose or proportions; the painting will be misaligned")
    n_raw = len(f)
    v, f = clean_mesh(v, f)
    v, f = decimate(v, f, faces)
    v, f = clean_mesh(v, f)
    say(f"mesh: {n_raw} -> {len(f)} faces")
    agree_geo = view_agreement(v, f, views)
    say("painting from the pictures")
    v2, f2, uv, n2, albedo, brep = bake(v, f, views, tex=tex, say=say)
    sc = float(height_m) / max(float(v2[:, 1].max() - v2[:, 1].min()), 1e-9)
    verts_m = (v2 - [0.0, v2[:, 1].min(), 0.0]) * sc
    report = {
        "name": name, "mode": "paint-supplied-mesh" if base_mesh is not None else "carve-from-outlines",
        "views": sorted(views), "faces": int(len(f2)), "vertices": int(len(v2)),
        "height_m": float(height_m), "size_m": [round(float(x), 4) for x in (verts_m.max(0) - verts_m.min(0))],
        "outline_iou": {k: round(x, 4) for k, x in agree_geo.items()},
        "mesh_registration": reg, "register": {k: rv.info for k, rv in views.items()}, "bake": brep, "notes": notes,
        "seconds": round(time.time() - t0, 1), "unlit": bool(unlit), "watertight_before_unwrap": _is_closed(f),
    }
    res = MVResult(verts_m, f2, uv, n2, albedo, report)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        from ..mesh.textured import glb_textured_bytes

        res.glb = out / f"{name}.glb"
        res.glb.write_bytes(glb_textured_bytes(verts_m, f2, uv, albedo, normals=n2, name=name, roughness=0.8, unlit=unlit))
        Image.fromarray(albedo).save(out / f"{name}_albedo.png")
        Image.fromarray(turntable(verts_m, f2, uv, albedo)).save(out / f"{name}_turntable.png")
        (out / f"{name}_report.json").write_text(json.dumps(report, indent=1, default=float))
    return res


def _is_closed(f: np.ndarray) -> bool:
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return bool((c == 2).all())
