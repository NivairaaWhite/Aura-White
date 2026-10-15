"""Views: names -> camera angles, and the numpy cameras / renderers used to compare a mesh with pictures.

Frame (identical to glTF and to ``aura_white.asset.camera``): +y up, the front of the object faces +z, +x is to the
viewer's right when looking at the front.  A camera orbits the object at azimuth ``az`` degrees: its position is in
the direction ``(sin az, ., cos az)``, so  front = 0,  left = +90 (the object's own left side faces the camera),
back = 180,  right = -90 (270).  ``el`` is the elevation above the horizon.

All the numbers below are *defaults the engine verifies* - `controller.calibrate_labels` checks a generated mesh against
the supplied pictures and swaps left/right if the evidence says the generator used the opposite convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# name -> (azimuth, elevation)
VIEW_ANGLES: Dict[str, Tuple[float, float]] = {
    "front": (0.0, 0.0), "back": (180.0, 0.0), "left": (90.0, 0.0), "right": (-90.0, 0.0),
    "top": (0.0, 90.0), "bottom": (0.0, -90.0),
    "front_left": (45.0, 0.0), "front_right": (-45.0, 0.0), "back_left": (135.0, 0.0), "back_right": (-135.0, 0.0),
}
_ALIASES = {      # whole words only; single letters and loose words ("b", "up", "face") are deliberately NOT here
    "rear": "back", "behind": "back", "posterior": "back", "reverse": "back", "backside": "back",
    "side": "left", "profile": "left", "lateral": "left",
    "above": "top", "overhead": "top", "under": "bottom", "below": "bottom",
    "frontleft": "front_left", "leftfront": "front_left", "frontright": "front_right", "rightfront": "front_right",
    "backleft": "back_left", "leftback": "back_left", "backright": "back_right", "rightback": "back_right",
    "34": "front_left", "threequarter": "front_left", "ref": "front", "reference": "front", "main": "front",
}
_TOKEN = re.compile(r"[a-z]+|\d+")


def canonical_label(name: str) -> Optional[str]:
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    n = _ALIASES.get(n, n)
    return n if n in VIEW_ANGLES else None


def infer_label(path_or_name: str) -> Optional[str]:
    """'knight_back.png' -> 'back'.  Looks at whole tokens only ('feedback' does not contain 'back')."""
    stem = Path(str(path_or_name)).stem.lower()
    toks = _TOKEN.findall(stem.replace("-", " ").replace("_", " "))
    joined = [a + b for a, b in zip(toks, toks[1:])]
    for cand in toks + joined:                                   # single tokens first, then neighbouring pairs
        lab = canonical_label(cand)
        if lab:
            return lab
    return None


@dataclass(frozen=True)
class ViewSpec:
    label: str
    az: float
    el: float = 0.0

    @staticmethod
    def parse(text: str) -> "ViewSpec":
        """'back', 'left@90', 'quarter@35,20' -> ViewSpec (explicit angles win over names)."""
        name, _, ang = text.partition("@")
        lab = canonical_label(name)
        az, el = VIEW_ANGLES[lab] if lab else (None, None)
        if ang:
            parts = [p for p in re.split(r"[,/ ]+", ang.strip()) if p]
            az = float(parts[0])
            el = float(parts[1]) if len(parts) > 1 else (el if el is not None else 0.0)
        if az is None:
            raise ValueError(f"cannot tell the angle of view {text!r}: use one of {', '.join(VIEW_ANGLES)} or name@azimuth[,elevation]")
        return ViewSpec(lab or name.strip().lower() or f"az{az:g}", float(az), float(el or 0.0))


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def proper_rotations() -> List[np.ndarray]:
    """The 24 rotations that map the coordinate axes to themselves (generators disagree on which way is up/front)."""
    out = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in np.ndindex(2, 2, 2):
            R = np.zeros((3, 3))
            for i, p in enumerate(perm):
                R[i, p] = -1.0 if signs[i] else 1.0
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


# ------------------------------------------------------------------------------------------------ the camera
@dataclass(frozen=True)
class ViewCam:
    """Orthographic camera.  x_px = size/2 + scale*(x' - ox);  y_px = size/2 - scale*(sy - oy)  (pixel centres at +0.5)."""
    az: float
    el: float = 0.0
    size: int = 192
    scale: float = 1.0
    ox: float = 0.0
    oy: float = 0.0

    def _frame(self, P: np.ndarray):
        P = np.asarray(P, np.float64)
        a = np.radians(self.az)
        ca, sa = np.cos(a), np.sin(a)
        xr = P[:, 0] * ca - P[:, 2] * sa
        zr = P[:, 0] * sa + P[:, 2] * ca
        e = np.radians(self.el)
        sy = P[:, 1] * np.cos(e) - zr * np.sin(e)
        t = P[:, 1] * np.sin(e) + zr * np.cos(e)
        return xr, sy, t

    def project(self, P: np.ndarray):
        """World (N,3) -> (x_px, y_px, t) with t = distance towards the camera (larger = nearer)."""
        xr, sy, t = self._frame(P)
        return (self.size / 2.0 + self.scale * (xr - self.ox), self.size / 2.0 - self.scale * (sy - self.oy), t)

    def extents(self, P: np.ndarray):
        xr, sy, _ = self._frame(P)
        return xr.min(), xr.max(), sy.min(), sy.max()

    def with_(self, **kw) -> "ViewCam":
        d = dict(az=self.az, el=self.el, size=self.size, scale=self.scale, ox=self.ox, oy=self.oy)
        d.update(kw)
        return ViewCam(**d)


def light_mesh(v: np.ndarray, f: np.ndarray, max_faces: int = 24_000):
    """A cheap stand-in for scoring renders (vertex clustering keeps silhouettes within a pixel or two)."""
    if len(f) <= max_faces:
        return np.asarray(v, np.float64), np.asarray(f, np.int64)
    from ..asset.meshio import simplify
    sv, sf = simplify(np.asarray(v, np.float64), np.asarray(f, np.int64), int(max_faces))
    return np.asarray(sv, np.float64), np.asarray(sf, np.int64)


def render_maps(verts: np.ndarray, faces: np.ndarray, cam: ViewCam):
    """(mask bool (S,S), t float32 (S,S) NaN = empty, tri int64 (S,S), bary (S,S,3)) of a mesh seen from `cam`."""
    from ..asset.camera import rasterize_ortho

    x, y, t = cam.project(verts)
    tri, bary, depth = rasterize_ortho(np.stack([x, y], 1), t, np.asarray(faces, np.int64), cam.size, cam.size,
                                       nearest_is_larger=True)
    return tri >= 0, depth, tri, bary


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def square_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Pad a mask to a square (centred) and resample it to size x size."""
    from PIL import Image

    m = np.asarray(mask) > 0.5
    h, w = m.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = m * 255
    if s != size:
        sq = np.asarray(Image.fromarray(sq).resize((size, size), Image.BILINEAR if s > size else Image.NEAREST))
    return sq > 127


def mask_bbox(mask: np.ndarray, min_share: float = 2e-4):
    """(x0, x1, y0, y1) of the object, ignoring specks smaller than `min_share` of the mask's area."""
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    try:
        from scipy import ndimage as ndi
        lab, n = ndi.label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            keep = np.flatnonzero(sizes >= max(1.0, min_share * m.sum())) + 1
            m = np.isin(lab, keep)
    except Exception:
        pass
    ys, xs = np.nonzero(m)
    return xs.min(), xs.max() + 1, ys.min(), ys.max() + 1


def refine_cam(verts: np.ndarray, faces: np.ndarray, cam: ViewCam, target: np.ndarray, rounds: int = 5):
    """Coordinate descent on (scale, shift x, shift y) maximising silhouette IoU.  A bounding-box fit is dragged around by
    anything the mesh lacks (a dropped tail, a short arm); maximising the overlap aligns the bulk of the shape instead.
    Returns (cam, iou)."""
    best = mask_iou(render_maps(verts, faces, cam)[0], target)
    step_px, step_s = 0.03 * cam.size, 0.04
    for _ in range(rounds):
        improved = True
        guard = 0
        while improved and guard < 6:
            improved, guard = False, guard + 1
            for kind in ("s", "x", "y"):
                for sign in (1.0, -1.0):
                    if kind == "s":
                        cand = cam.with_(scale=cam.scale * (1.0 + sign * step_s))
                    elif kind == "x":
                        cand = cam.with_(ox=cam.ox + sign * step_px / cam.scale)
                    else:
                        cand = cam.with_(oy=cam.oy + sign * step_px / cam.scale)
                    iou = mask_iou(render_maps(verts, faces, cand)[0], target)
                    if iou > best + 1e-4:
                        cam, best, improved = cand, iou, True
        step_px *= 0.5
        step_s *= 0.5
    return cam, best


def fit_cam(verts: np.ndarray, faces: np.ndarray, az: float, el: float, target: np.ndarray, size: int = 192,
            refine: bool = True):
    """Place an orthographic camera so the mesh's silhouette lines up with `target`: a bounding-box initialisation (scale
    tried from the height, the width and their mean) followed by an IoU-maximising refinement.
    Returns (ViewCam, iou, target_mask_at_size)."""
    tm = square_mask(target, size)
    bb = mask_bbox(tm)
    base = ViewCam(az, el, size)
    if bb is None or len(verts) == 0:
        return base, 0.0, tm
    tx0, tx1, ty0, ty1 = bb
    tw, th, tcx, tcy = max(tx1 - tx0, 1), max(ty1 - ty0, 1), (tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0
    x0, x1, y0, y1 = base.extents(verts)
    mw, mh = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    starts = []
    for sc in sorted({th / mh, tw / mw, float(np.sqrt(th * tw / (mh * mw)))}):
        cam = ViewCam(az, el, size, sc, cx - (tcx - size / 2.0) / sc, cy + (tcy - size / 2.0) / sc)
        starts.append((cam, mask_iou(render_maps(verts, faces, cam)[0], tm)))
    if not refine:
        return (*max(starts, key=lambda t: t[1]), tm)
    # a bounding box is dragged around by whatever the mesh lacks, so refine from every start and keep the best
    lv, lf = light_mesh(verts, faces, 12_000)
    best = (base, -1.0)
    for cam0, _ in starts:
        cam, iou = refine_cam(lv, lf, cam0, tm, rounds=4)
        if iou > best[1]:
            best = (cam, iou)
    return best[0], best[1], tm
