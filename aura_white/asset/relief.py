"""Rounded relief ("inflate") proxy: the reference view stays pixel-exact, the sides get real volume.

Every vertex sits on its own view ray, so the picture is reproduced exactly from the reference camera whatever depth
field we choose.  The depth field is a *rolling-ball* inflation of the silhouette:

  * ``d``      distance to the outline (px)
  * ``R``      radius of the largest inscribed disc covering the pixel, capped at ``rcap``
  * half-thickness  h = roundness * R * sqrt(u (2 - u)),  u = min(d / R, 1)       (a circular section)

so a 3-px chain becomes a round wire, a 30-px ribbon a flat tube, and a wide dress a plate with a generously rounded
edge instead of a razor-thin slab.  Optional priors for standing figures: a lens-shaped body section (thicker on the
centre line) and a gentle backwards bend of the outer parts (tails, sleeves, skirt hems).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..forge.patch import build_relief
from .camera import OrthoCamera
from .layers import Chart, Layer


@dataclass
class ReliefSpec:
    rcap: float = 0.04          # ball radius cap, as a fraction of the subject's longest side
    roundness: float = 0.55     # 1 = circular sections, smaller = flatter
    lens: float = 0.0           # 0..1 lens-shaped section for wide parts (character body)
    bend: float = 0.0           # 0..1 outer parts recede backwards (relative to half-width)
    smooth: float = 1.6         # px, smoothing of the thickness map
    back_ratio: float = 1.0     # back half-thickness relative to the front one
    plane: str = "facing"       # facing | upright | ground : the flat surface the relief is inflated from


def rolling_ball_radius(mask: np.ndarray, rcap: float, levels: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """(d, R): distance to the outline and the radius (<= rcap) of the largest inscribed disc covering each pixel."""
    from scipy import ndimage as ndi
    dt = ndi.distance_transform_edt(mask).astype(np.float32)
    top = float(min(dt.max(), max(rcap, 1.0)))
    radii = np.unique(np.round(np.geomspace(1.0, max(top, 1.0), levels), 2))[::-1]
    R = np.zeros_like(dt)
    for r in radii:
        core = dt >= r
        if not core.any():
            continue
        covered = ndi.distance_transform_edt(~core) <= r
        R = np.where((R == 0) & covered & mask, r, R)
    R = np.where(mask & (R == 0), np.maximum(dt, 1.0), R)
    return dt, np.minimum(R, max(rcap, 1.0))


def _runs_position(mask: np.ndarray) -> np.ndarray:
    """Position of each masked pixel inside its horizontal run, in [-1, 1] (0 = run centre)."""
    H, W = mask.shape
    idx = np.broadcast_to(np.arange(W), (H, W))
    prev = np.pad(mask, ((0, 0), (1, 0)))[:, :-1]
    start = np.where(mask & ~prev, idx, -1)
    left = idx - np.maximum.accumulate(start, axis=1)
    m2 = mask[:, ::-1]
    prev2 = np.pad(m2, ((0, 0), (1, 0)))[:, :-1]
    idx2 = np.broadcast_to(np.arange(W), (H, W))
    start2 = np.where(m2 & ~prev2, idx2, -1)
    right = (idx2 - np.maximum.accumulate(start2, axis=1))[:, ::-1]
    tot = np.maximum(left + right, 1)
    return np.where(mask, (left - right) / tot, 0.0).astype(np.float32)


def thickness_fields(alpha: np.ndarray, spec: ReliefSpec, longest: float):
    """-> (half_front, half_back, z_mid) maps in px."""
    from scipy import ndimage as ndi
    mask = alpha >= 0.5
    rcap = max(1.0, spec.rcap * longest)
    d, R = rolling_ball_radius(mask, rcap)
    u = np.clip(d / np.maximum(R, 1e-3), 0.0, 1.0)
    h = spec.roundness * R * np.sqrt(u * (2.0 - u))
    if spec.lens > 0:
        s = _runs_position(mask)
        sat = np.clip((R - 0.8 * rcap) / (0.4 * rcap), 0.0, 1.0)      # only where the ball is saturated (wide parts)
        h = h * (1.0 - spec.lens * sat * (1.0 - np.sqrt(np.clip(1.0 - s * s, 0.0, 1.0))))
    if spec.smooth > 0:
        w = ndi.gaussian_filter(mask.astype(np.float32), spec.smooth)
        h = np.where(mask, ndi.gaussian_filter(h * mask, spec.smooth) / np.maximum(w, 1e-3), 0.0)
    h = np.where(mask, h, 0.0).astype(np.float32)
    zmid = np.zeros_like(h)
    if spec.bend > 0:
        s = _runs_position(mask)
        # per row: the whole silhouette's half-width, centre on the symmetry axis of that row
        H, W = mask.shape
        cols = np.arange(W, dtype=np.float32)[None, :]
        anym = mask.any(1)
        lo = np.where(anym, np.argmax(mask, 1), 0).astype(np.float32)
        hi = np.where(anym, W - 1 - np.argmax(mask[:, ::-1], 1), 0).astype(np.float32)
        centre = 0.5 * (lo + hi)
        half = np.maximum(0.5 * (hi - lo), 1.0)
        x = np.clip(np.abs(cols - centre[:, None]) / half[:, None], 0.0, 1.0)
        zmid = (-spec.bend * half[:, None] * (1.0 - np.sqrt(1.0 - np.minimum(x, 0.999) ** 2))).astype(np.float32)
        zmid = np.where(mask, zmid, 0.0)
    return h, (h * spec.back_ratio).astype(np.float32), zmid


def build_relief_layer(alpha: np.ndarray, cam: OrthoCamera, spec: ReliefSpec, chart_front: Chart, chart_back: Optional[Chart],
                       crop: Tuple[int, int, int, int], longest: float, role: str = "body", name: str = "relief",
                       lod: float = 1.0, max_faces: int = 0, base_t: Optional[np.ndarray] = None,
                       region: Optional[np.ndarray] = None) -> Optional[Layer]:
    """Mesh the subject (or `region`, a boolean sub-mask) as a closed rounded relief.
    `base_t` (H,W) is an extra mid-surface depth (an upright / ground plane ramp, or a proxy surface to hug)."""
    a = np.asarray(alpha, np.float32)
    if region is not None:
        a = np.where(region, a, 0.0)
    hf, hb, zmid = thickness_fields(a, spec, longest)
    if base_t is not None:
        zmid = zmid + base_t
    rm = build_relief(a, hf, hb, zmid, boundary_step=1.5 * lod, interior_step=(2.5 * lod + 1.5, 48.0 * lod),
                      grade=0.30, simplify_tol=0.22 * lod, edge_half=0.15, max_faces=max_faces)
    if rm is None:
        return None
    n = rm.n_surface
    P = rm.verts.astype(np.float64)                          # x_px, y_px, z (toward the viewer)
    world = cam.unproject(P[:, 0], P[:, 1], P[:, 2])
    faces = rm.faces
    # image space is left-handed relative to the world: make the outside the outside again
    v = world
    vol = float(np.sum(np.einsum("ij,ij->i", v[faces[:, 0]], np.cross(v[faces[:, 1]], v[faces[:, 2]]))) / 6.0)
    if vol < 0:
        faces = faces[:, [0, 2, 1]]
    x0, y0, x1, y1 = crop
    xy = rm.xy.astype(np.float64)
    front_uv = np.stack([xy[:, 0] - x0, xy[:, 1] - y0], 1)
    back_uv = np.stack([x1 - xy[:, 0], xy[:, 1] - y0], 1)       # mirrored: the back is the picture seen from behind
    vert_uv = np.concatenate([front_uv, back_uv], 0)
    is_back = faces >= n
    nb = is_back.sum(1)
    if chart_back is None:        # skin patches: the back shows the same picture again
        chart = np.full(len(faces), chart_front.cid, np.int32)
        cuv = front_uv[np.where(faces >= n, faces - n, faces)]
    else:
        chart = np.where(nb == 3, chart_back.cid, chart_front.cid).astype(np.int32)   # walls (mixed) use the front chart
        cuv = np.where((nb == 3)[:, None, None], vert_uv[faces], front_uv[np.where(faces >= n, faces - n, faces)])
    return Layer(name, world, faces, cuv.astype(np.float32), chart, role=role,
                 info=dict(front_vertices=int(n), max_half_thickness=float(hf.max()), rcap_px=float(spec.rcap * longest)))
