"""Proxy builders: turn a Subject into charts + (per level of detail) layers.

Every builder owns the chart layout (identical for all LODs, so one atlas serves the whole LOD chain) and produces the
meshes for a given LOD scale through ``layers(lod)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .camera import OrthoCamera
from .layers import Chart, Layer, merge_layers
from .relief import ReliefSpec, build_relief_layer
from .solids import disc, revolve, uv_sphere
from .subject import Subject

# LOD level -> mesh density factors.  Level 0 is the full-detail asset.
SPHERE_SEGMENTS = {0: (96, 48), 1: (64, 32), 2: (32, 16), 3: (16, 8)}


class Builder:
    kind = "base"
    tier = "B"          # A analytic (3-D consistent), B relief / proxy (volume added, hidden sides plausible)

    def __init__(self, subject: Subject, cam: OrthoCamera, opts: Dict[str, object]):
        self.s, self.cam, self.opts = subject, cam, dict(opts)
        self.charts: Dict[int, Chart] = {}
        self.notes: List[str] = []
        self.info: Dict[str, object] = {}

    @staticmethod
    def camera(subject: Subject, opts: Dict[str, object]) -> OrthoCamera:
        W, H = subject.size
        return OrthoCamera(W, H, float(opts.get("elev") or 0.0))

    def layers(self, lod: int = 0, max_faces: int = 0) -> List[Layer]:
        raise NotImplementedError

    def collision_groups(self, layers: List[Layer]) -> List[np.ndarray]:
        """Point sets (world px) whose convex hulls make the collision proxy.  Default: k-means clusters."""
        from .collision import kmeans_hulls
        pts = np.concatenate([L.verts for L in layers])
        return kmeans_hulls(pts, int(self.opts.get("collision_hulls", 8)))

    def _chart(self, *a, **k) -> Chart:
        ch = Chart(len(self.charts), *a, **k)
        self.charts[ch.cid] = ch
        return ch


class SphereBuilder(Builder):
    kind = "sphere"
    tier = "A"

    def collision_groups(self, layers):
        return [np.concatenate([L.verts for L in layers])]

    def __init__(self, subject, cam, opts):
        super().__init__(subject, cam, opts)
        cx, cy, rx, ry = subject.disc
        rz = 0.5 * (rx + ry)
        self.centre = cam.unproject(cx, cy, 0.0)
        self.radii = (rx * 0.9985, ry * 0.9985, rz)           # a hair inside the matte: the limb never shows background
        W = int(round(2 * np.pi * rz))
        H = int(round(np.pi * rz))
        self.chart = self._chart(W, H, "param", fill="through", density=float(opts.get("density", 1.0)),
                                 params={"reflect": True, "trust_min": float(opts.get("trust_min", 0.2))})
        self.rim_inset = float(opts.get("rim_inset", 2.0))
        self.info.update(radii_px=[float(r) for r in self.radii], rim_inset_px=self.rim_inset)
        self.notes.append("sphere: the far hemisphere shows the picture mirrored through the ball")
        if self.rim_inset > 0:
            self.notes.append(f"sphere: the outer {self.rim_inset:g} px of the limb are re-grown from the colours just inside "
                              "(limb darkening in the picture would otherwise show as a dark stripe when the ball turns; "
                              "pass rim_inset=0 to keep the picture's rim exactly)")

    def source_alpha(self):
        """Alpha used to decide which picture pixels are trusted as colour (the rim ring is not)."""
        if self.rim_inset <= 0:
            return self.s.alpha
        from scipy import ndimage as ndi
        r = self.rim_inset
        k = int(np.ceil(r))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disk = (xx ** 2 + yy ** 2) <= r * r
        return ndi.binary_erosion(self.s.alpha >= 0.5, structure=disk).astype(np.float32)

    def layers(self, lod=0, max_faces=0):
        nl, nt = SPHERE_SEGMENTS[min(lod, 3)]
        return [uv_sphere(self.centre, self.radii, nl, nt, self.chart, "moon")]


# ------------------------------------------------------------------------------------------------ arena
@dataclass
class RingFit:
    xc: float            # symmetry axis (image column)
    R: float             # outer radius (px)
    elev: float          # camera elevation (deg)
    yb0: float           # image row of the base ellipse centre (wall bottom, ground level)
    yt0: float           # image row of the rim-top ellipse centre
    wall_h: float        # wall height in world px
    residual: float      # rms of the bottom-arc fit, px
    floor_r: float = 0.0
    floor_y: float = 0.0


def robust_line(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y ~ a + b q with outliers (spires, stairs) down-weighted."""
    w = np.ones(len(q))
    sol = np.zeros(2)
    for _ in range(12):
        A = np.c_[np.ones(len(q)), q] * w[:, None]
        sol = np.linalg.lstsq(A, y * w, rcond=None)[0]
        res = y - (sol[0] + sol[1] * q)
        sg = max(1.4826 * np.median(np.abs(res - np.median(res))), 1.0)
        w = (np.abs(res) < 2.0 * sg).astype(float) + 1e-3
    return sol


def fit_ring(alpha: np.ndarray, elev: Optional[float] = None, wall_ratio: float = 0.36, trim: float = 0.9) -> RingFit:
    """Fit a vertical cylinder to the silhouette of a ring-like structure seen from above.

    Orthographic model: the bottom arc is the front half of the base ellipse, the top arc the back half of the rim
    ellipse.  Illustrations are drawn with a little perspective, which makes the near wall look taller than the far wall,
    so the wall height cannot be read from the two arcs alone.  Instead the wall height is a ratio of the radius
    (`wall_ratio`, 0.36 R is typical for amphitheatres) and the elevation is solved from the silhouette's vertical extent
    H cos(e) + 2 R sin(e) = extent, which puts the front rim edge where the artwork's facade top really is.
    """
    hard = alpha >= 0.5
    ys, xs = np.nonzero(hard)
    x0, x1 = xs.min(), xs.max() + 1
    xc, R = 0.5 * (x0 + x1), 0.5 * (x1 - x0) - 0.5
    # the wall's side lines are vertical: take the typical half-width over the middle rows (towers and bastions that
    # stick out at the extremes are left to the skin patches)
    row_has = hard.any(1)
    y_lo, y_hi = int(ys.min()), int(ys.max())
    mid_rows = [y for y in range(y_lo + int(0.35 * (y_hi - y_lo)), y_lo + int(0.8 * (y_hi - y_lo))) if row_has[y]]
    if mid_rows:
        left = np.array([np.nonzero(hard[y])[0].min() for y in mid_rows], np.float64)
        right = np.array([np.nonzero(hard[y])[0].max() + 1 for y in mid_rows], np.float64)
        Rm = 0.5 * float(np.median(right - left)) - 0.5
        if 0.9 * R < Rm <= R:
            xc, R = 0.5 * float(np.median(left + right)), Rm
    cols = np.arange(int(np.ceil(xc - trim * R)), int(np.floor(xc + trim * R)) + 1)
    cols = cols[hard[:, cols].any(0)]
    ybot = np.array([np.nonzero(hard[:, c])[0].max() + 1.0 for c in cols])
    ytop = np.array([np.nonzero(hard[:, c])[0].min() + 0.0 for c in cols])
    q = np.sqrt(np.maximum(R * R - (cols + 0.5 - xc) ** 2, 0.0))

    sb = robust_line(q, ybot)
    st = robust_line(q, ytop)
    Yb_peak = float(sb[0] + sb[1] * R)               # lowest point of the bottom arc (centre column)
    Yt_peak = float(st[0] - abs(st[1]) * R)          # highest point of the top arc (spires ignored by the robust fit)
    extent = Yb_peak - Yt_peak
    H = wall_ratio * R
    if elev is None:
        lo, hi = np.radians(12.0), np.radians(55.0)
        e = lo
        for _ in range(60):                          # bisection on  H cos e + 2 R sin e - extent
            mid = 0.5 * (lo + hi)
            val = H * np.cos(mid) + 2 * R * np.sin(mid)
            if val < extent:
                lo = mid
            else:
                hi = mid
        e = float(np.degrees(0.5 * (lo + hi)))
    else:
        e = float(elev)
    k, c = np.sin(np.radians(e)), np.cos(np.radians(e))
    yb0 = Yb_peak - R * k
    yt0 = yb0 - H * c
    res_rms = float(np.sqrt(np.mean((ybot - (sb[0] + sb[1] * q)) ** 2)))
    return RingFit(float(xc), float(R), e, float(yb0), float(yt0), float(H), res_rms, 0.72 * R, 0.55 * H)


def nearest_fill(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    if valid.all() or not valid.any():
        return np.where(valid, field, 0.0)
    _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
    return field[iy, ix]


def snap_to_matte(layer: Layer, cam: OrthoCamera, alpha: np.ndarray, inset: float = 0.5) -> Layer:
    """Slide every vertex that lies outside the silhouette to the nearest point inside it, *in the picture plane* (its
    depth along the view ray is kept), so the proxy's outline cannot stick out of the matte in the reference view."""
    from scipy import ndimage as ndi
    hard = alpha >= 0.5
    if not hard.any():
        return layer
    H, W = hard.shape
    px, py, t = cam.project(layer.verts)
    ix = np.clip(np.floor(px).astype(np.int64), 0, W - 1)
    iy = np.clip(np.floor(py).astype(np.int64), 0, H - 1)
    outside = ~hard[iy, ix] | (px < 0) | (px > W) | (py < 0) | (py > H)
    if not outside.any():
        return layer
    _, (ny, nx) = ndi.distance_transform_edt(~hard, return_indices=True)
    tx = nx[iy[outside], ix[outside]] + 0.5
    ty = ny[iy[outside], ix[outside]] + 0.5
    # step a little further in so the vertex sits just inside the outline
    dx, dy = tx - px[outside], ty - py[outside]
    nrm = np.maximum(np.hypot(dx, dy), 1e-9)
    tx = tx + inset * dx / nrm
    ty = ty + inset * dy / nrm
    v = layer.verts.copy()
    v[outside] = cam.unproject(tx, ty, t[outside])
    return Layer(layer.name, v, layer.faces, layer.corner_uv, layer.face_chart, layer.role,
                 dict(layer.info, snapped_vertices=int(outside.sum())), layer.face_side)


def trim_to_matte(layer: Layer, cam: OrthoCamera, alpha: np.ndarray, margin: float = 2.0) -> Layer:
    """Drop proxy faces that lie wholly outside the silhouette (the proxy may overshoot the matte by a few px)."""
    from scipy import ndimage as ndi
    hard = alpha >= 0.5
    d_out = ndi.distance_transform_edt(~hard)
    px, py, _ = cam.project(layer.verts)
    H, W = hard.shape
    ix = np.clip(px.astype(np.int64), 0, W - 1)
    iy = np.clip(py.astype(np.int64), 0, H - 1)
    outside = (d_out[iy, ix] > margin) | (px < -margin) | (px > W + margin) | (py < -margin) | (py > H + margin)
    keep = ~outside[layer.faces].all(1)
    if keep.all():
        return layer
    return Layer(layer.name, layer.verts, layer.faces[keep], layer.corner_uv[keep], layer.face_chart[keep], layer.role,
                 dict(layer.info, trimmed_faces=int((~keep).sum())), layer.sides()[keep])


def skin_base_depth(t: np.ndarray, covered: np.ndarray, cam: OrthoCamera) -> np.ndarray:
    """Depth of the surface a patch of 'extra' pixels stands on: a patch rises like an upright wall from the nearest proxy
    surface *below it in the same picture column* (a spire from the roof it stands on); where there is nothing below, the
    nearest proxy depth in the picture is used."""
    H, W = covered.shape
    rows = np.broadcast_to(np.arange(H)[:, None], (H, W))
    below = np.where(covered, rows, 10 * H)
    below = np.minimum.accumulate(below[::-1], axis=0)[::-1]            # first covered row at or below each pixel
    has = below < H
    cols = np.broadcast_to(np.arange(W)[None, :], (H, W))
    tc = np.where(has, t[np.minimum(below, H - 1), cols], 0.0)
    base = tc + np.tan(cam.e) * (below - rows)                         # upright wall: dt/d(screen-up) = tan e
    near = nearest_fill(np.where(covered, t, 0.0).astype(np.float32), covered)
    return np.where(has, base, near).astype(np.float32)


def skin_layer(builder: "Builder", body_layers: List[Layer], min_area: int = 24, name: str = "skin",
               extra: Optional[np.ndarray] = None, extra_base: Optional[np.ndarray] = None,
               spec: Optional[ReliefSpec] = None) -> Optional[Layer]:
    """Everything the artwork shows that the proxy body does not cover (spires, stairs, ornaments ...): thin relief
    patches that hug the nearest proxy surface, so the silhouette of the reference view is complete."""
    from .camera import render_view
    from .layers import merge_layers
    from scipy import ndimage as ndi
    s, cam = builder.s, builder.cam
    m = merge_layers(body_layers)
    _, _, t = render_view(cam, m.verts, m.faces)
    covered = np.isfinite(t)
    hard = s.alpha >= 0.5
    resid = hard & ~ndi.binary_dilation(covered, iterations=1)
    if extra is not None:
        resid = resid | (hard & extra)
    lab, n = ndi.label(resid, structure=np.ones((3, 3)))
    if n == 0:
        return None
    area = np.bincount(lab.ravel())
    keep = area >= min_area
    keep[0] = False
    resid = keep[lab]
    if not resid.any():
        return None
    base = skin_base_depth(t, covered, cam)
    if extra is not None and extra_base is not None:
        base = np.where(extra, extra_base, base).astype(np.float32)
    ys, xs = np.nonzero(resid)
    pad = 3
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + 1 + pad, s.size[0])
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + 1 + pad, s.size[1])
    ch = builder._chart(x1 - x0, y1 - y0, "front", crop=(x0, y0, x1, y1), density=float(builder.opts.get("skin_density", 0.6)))
    longest = max(s.metrics["width"], s.metrics["height"])
    spec = spec or ReliefSpec(rcap=0.004, roundness=0.7, smooth=0.8)
    alpha = np.where(resid, np.maximum(s.alpha, 0.5), 0.0).astype(np.float32)
    builder._skin_ctx = dict(chart=ch, alpha=alpha, base=base, crop=(x0, y0, x1, y1), spec=spec, longest=longest,
                             name=name, pixels=int(resid.sum()))
    L = skin_build(builder, 0)
    if L is not None:
        builder.notes.append(f"skin: {int(resid.sum())} px the proxy body does not cover were added as relief patches")
    return L


def skin_build(builder: "Builder", lod: int = 0) -> Optional[Layer]:
    """The skin patches at a level of detail (same chart for every level)."""
    c = getattr(builder, "_skin_ctx", None)
    if not c:
        return None
    lod = min(int(lod), 3)
    budget = max(int(int(builder.opts.get("skin_faces", 16000)) * LOD_FACE_FRACTION[lod]), 600)
    L = build_relief_layer(c["alpha"], builder.cam, c["spec"], c["chart"], None, c["crop"], c["longest"], role="skin",
                           name=c["name"], lod=LOD_STEP[lod], base_t=c["base"], max_faces=budget)
    if L is not None:
        L.info["skin_pixels"] = c["pixels"]
    return L


def ring_contours(alpha: np.ndarray, xc: float, R: float, window: int = 201):
    """Smooth bottom / top contour of a ring silhouette as functions of the image column: the robust ellipse arc plus a
    heavily median-filtered residual (so spires and stairs do not pull the footprint around, but real asymmetry does)."""
    from scipy import ndimage as ndi
    hard = alpha >= 0.5
    H, W = hard.shape
    cols = np.arange(W)
    has = hard.any(0)
    ybot = np.where(has, H - np.argmax(hard[::-1], 0).astype(np.float64), np.nan)
    ytop = np.where(has, np.argmax(hard, 0).astype(np.float64), np.nan)
    sel = has & (np.abs(cols + 0.5 - xc) < 0.92 * R)
    q = np.sqrt(np.maximum(R * R - (cols + 0.5 - xc) ** 2, 0.0))
    out = []
    for y in (ybot, ytop):
        a, b = robust_line(q[sel], y[sel])
        fit = a + b * q
        res = np.where(sel, y - fit, 0.0)
        res = ndi.median_filter(res, size=window, mode="nearest")
        res = ndi.gaussian_filter1d(res, 8.0, mode="nearest")
        res = np.where(np.abs(cols + 0.5 - xc) < 0.95 * R, res, res[np.argmin(np.abs(cols + 0.5 - xc - 0.95 * R))])
        out.append(fit + res)
    return cols + 0.5, out[0], out[1]


class ArenaBuilder(Builder):
    """Ring structure (amphitheatre, arena, stadium, colosseum) seen from above: outer wall + rim + stands + floor.

    The footprint is a circle warped (in depth only) so that, in the reference view, the base of the wall lies exactly on
    the bottom outline of the matte and the rim of the far wall on its top outline - the artwork has some perspective,
    a plain cylinder would not fit it."""
    kind = "arena"
    tier = "A/B"

    def collision_groups(self, layers):
        from .collision import sector_points
        c = self.centre
        wedge_pts = np.concatenate([L.verts for L in layers if L.name in ("arena_wall", "arena_stands", "arena_rim")])
        groups = sector_points(wedge_pts, c, int(self.opts.get("collision_wedges", 24)))
        fl = next(L for L in layers if L.name == "arena_floor")
        top = fl.verts.copy()
        bot = fl.verts.copy()
        bot[:, 1] = 0.0
        groups.append(np.concatenate([top, bot]))                     # the floor: a solid cylinder under the arena floor
        return groups

    @staticmethod
    def camera(subject: Subject, opts: Dict[str, object]) -> OrthoCamera:
        fit = fit_ring(subject.alpha, opts.get("elev"), float(opts.get("wall_ratio", 0.36)))
        W, H = subject.size
        return OrthoCamera(W, H, fit.elev)

    def __init__(self, subject, cam, opts):
        super().__init__(subject, cam, opts)
        fit = fit_ring(subject.alpha, opts.get("elev"), float(opts.get("wall_ratio", 0.36)))
        self.fit = fit
        R, H = fit.R * 0.999, fit.wall_h
        fl = opts.get("floor_ratio")                       # (radius / R, depth below the rim / wall height)
        r_f = (fl[0] if fl else 0.70) * R
        y_f = H * (1.0 - (fl[1] if fl else 0.42))
        wr = 0.06 * R
        self.R, self.H, self.r_f, self.y_f, self.wr = R, H, r_f, y_f, wr
        base = cam.ground_point(fit.xc, fit.yb0, 0.0)
        self.centre = (float(base[0]), float(base[2]))
        self._make_warp(subject.alpha, cam, fit, R, H)
        outer = np.stack([np.full(6, R), np.linspace(0.0, H, 6)], 1)
        stands = np.stack([np.linspace(R - wr, r_f, 8), np.linspace(H, y_f, 8)], 1)
        self.p_outer, self.p_rim, self.p_stands = outer, np.array([[R, H], [R - wr, H]]), stands
        lo = float(np.hypot(np.diff(outer[:, 0]), np.diff(outer[:, 1])).sum())
        ls = float(np.hypot(np.diff(stands[:, 0]), np.diff(stands[:, 1])).sum())
        dens = float(opts.get("density", 1.0))
        half = int(round(np.pi * R))
        # the outer wall is seen on one side only (|theta| < 90 deg), so one half-turn chart serves both halves; the stands
        # are seen over most of the circle (both sides at the left and right), so they get a full-turn chart
        self.wall_chart = self._chart(half, int(np.ceil(lo)) + 1, "param", fill="fold2", density=dens,
                                      params={"wrap_u": True})
        self.stands_chart = self._chart(int(round(2 * np.pi * R)), int(np.ceil(ls)) + 1, "param", fill="rot180",
                                        density=float(opts.get("stands_density", 0.8 * dens)), params={"wrap_u": True})
        self.rim_chart = self._chart(int(round(2 * np.pi * R)), int(round(wr)) + 1, "param", fill="nearest", density=dens,
                                     params={"wrap_u": True})
        self.floor_chart = self._chart(int(round(2 * r_f)), int(round(2 * r_f)), "param", fill="planar180",
                                       density=float(opts.get("floor_density", dens)))
        self.base_chart = self._chart(32, 32, "param", fill="nearest", density=0.25)
        self.info.update(elev=fit.elev, R_px=R, wall_h_px=H, floor_r_px=r_f, floor_depth_px=H - y_f, fit_rms_px=fit.residual,
                         footprint_depth_scale=[float(self._s.min()), float(self._s.max())])
        self.notes.append(f"arena: camera elevation {fit.elev:.1f} deg, outer radius {R:.0f} px, wall {H:.0f} px "
                          f"(wall height is assumed to be {opts.get('wall_ratio', 0.36)} x radius; pass wall_ratio= to change it)")
        self.notes.append("arena: the footprint follows the picture's outline; the sides the picture cannot see are copies "
                          "of the visible side turned half a revolution (one texture serves both halves)")

    # ---- depth warp of the footprint
    def _make_warp(self, alpha, cam, fit, R, H):
        xc = fit.xc
        cols, ybot, ytop = ring_contours(alpha, xc, R)
        th = np.linspace(-np.pi, np.pi, 1441)
        xp = xc + R * np.sin(th)
        cz = self.centre[1]
        yb = np.interp(xp, cols, ybot)
        yt = np.interp(xp, cols, ytop)
        zf = cam.ground_points(xp, yb, 0.0)[:, 2]
        zb = cam.ground_points(xp, yt, H)[:, 2]
        cth = np.cos(th)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(cth > 0, (zf - cz) / (R * cth), (zb - cz) / (R * cth))
        ok = np.abs(cth) > 0.3
        ok &= np.isfinite(s) & (s > 0.5) & (s < 1.6)
        s_full = np.interp(th, th[ok], s[ok], period=2 * np.pi)
        self._th, self._s = th, s_full

    def warp(self, layer: Layer) -> Layer:
        cx, cz = self.centre
        v = layer.verts.copy()
        th = np.arctan2(v[:, 0] - cx, v[:, 2] - cz)
        v[:, 2] = cz + (v[:, 2] - cz) * np.interp(th, self._th, self._s, period=2 * np.pi)
        return Layer(layer.name, v, layer.faces, layer.corner_uv, layer.face_chart, layer.role, layer.info, layer.face_side)

    def _body_layers(self, lod: int) -> List[Layer]:
        nth = {0: 160, 1: 96, 2: 48, 3: 24}[min(lod, 3)]
        nr = {0: 12, 1: 8, 2: 4, 3: 2}[min(lod, 3)]
        c = self.centre
        outer = revolve(self.p_outer, nth, c, self.wall_chart, name="arena_wall", fold=True)
        stands = revolve(self.p_stands, nth, c, self.stands_chart, name="arena_stands")
        rim = revolve(self.p_rim, nth, c, self.rim_chart, name="arena_rim")
        fl = disc((c[0], self.y_f, c[1]), self.r_f, nth, nr, self.floor_chart, up=True, name="arena_floor")
        bs = disc((c[0], 0.0, c[1]), self.R, nth, 2, self.base_chart, up=False, name="arena_base")
        return [trim_to_matte(snap_to_matte(self.warp(L), self.cam, self.s.alpha), self.cam, self.s.alpha, margin=8.0)
                for L in (outer, stands, rim, fl, bs)]

    def _towers(self, body: List[Layer]):
        """Towers and other tall things standing on the near rim rise in front of the floor in the picture and would be
        painted onto it (long dark spikes when the arena turns).  They are recognised as busy (high texture energy)
        regions in the floor part of the picture that are attached to the near wall, and stand up as relief instead.
        -> (mask, base depth) or (None, None)."""
        from scipy import ndimage as ndi
        from .camera import render_view
        s, cam = self.s, self.cam

        def cover(names):
            m = merge_layers([L for L in body if L.name in names])
            _, _, t = render_view(cam, m.verts, m.faces)
            return np.isfinite(t), t

        fl, tf = cover(("arena_floor",))
        near, tn = cover(("arena_wall", "arena_rim"))
        allm = merge_layers(body)
        _, _, tall = render_view(cam, allm.verts, allm.faces)
        floor_vis = fl & np.isfinite(tall) & (np.abs(tall - tf) < 1.5)
        core = ndi.binary_erosion(floor_vis, iterations=40)
        if floor_vis.sum() < 5000 or not core.any():
            return None, None
        lum = (0.2126 * s.rgb[..., 0] + 0.7152 * s.rgb[..., 1] + 0.0722 * s.rgb[..., 2]).astype(np.float32)
        E = ndi.uniform_filter(ndi.gaussian_gradient_magnitude(lum, 1.5), 13)
        base_e = float(np.median(E[core]))
        cand = ndi.binary_opening(floor_vis & (E > 3.0 * base_e), structure=np.ones((5, 5)))
        rimband = ndi.binary_dilation(near, iterations=10) & floor_vis
        lab, n = ndi.label(ndi.binary_closing(cand, structure=np.ones((5, 5)), iterations=2))
        cy = float(cam.project(np.array([[self.centre[0], self.y_f, self.centre[1]]]))[1][0])
        ys = np.nonzero(floor_vis.any(1))[0]
        y_front = cy + 0.15 * (ys.max() - cy)                 # only the front part of the floor: that is where towers stand
        keep = np.zeros(n + 1, bool)
        for i in range(1, n + 1):
            comp = lab == i
            yy = np.nonzero(comp.any(1))[0]
            if comp.sum() > 400 and (comp & rimband).any() and yy.mean() > y_front:
                keep[i] = True
        mask = keep[lab]
        if mask.sum() < 300:
            return None, None
        mask = ndi.binary_dilation(mask, iterations=3) & (s.alpha >= 0.5)
        tb = np.where(near, tn, np.nan)
        base = skin_base_depth(np.nan_to_num(tb, nan=0.0), near, cam)       # stand on the near wall / rim below them
        self.notes.append(f"arena: {int(mask.sum())} picture pixels of towers in front of the floor stand up as relief")
        return mask, base

    def layers(self, lod=0, max_faces=0):
        body = self._body_layers(lod)
        if "_skin_ctx" not in self.__dict__:
            b0 = self._body_layers(0)
            mask, base = self._towers(b0)
            skin_layer(self, b0, extra=mask, extra_base=base,
                       spec=ReliefSpec(rcap=float(self.opts.get("skin_rcap", 0.015)), roundness=0.8, smooth=0.8))
            self.__dict__.setdefault("_skin_ctx", None)
        sk = skin_build(self, lod)
        return body + ([sk] if sk is not None else [])


# ------------------------------------------------------------------------------------------------ relief (character / prop)
RELIEF_PRESETS = {
    # rcap: ball radius cap (fraction of the longest side)   roundness: 1 = circular sections   lens: lens-shaped body section
    "character": dict(rcap=0.085, roundness=0.9, lens=0.45, smooth=1.6),
    "prop": dict(rcap=0.014, roundness=0.75, lens=0.0, smooth=1.0),
    "building": dict(rcap=0.12, roundness=0.8, lens=0.3, smooth=2.0),
}
LOD_FACE_FRACTION = {0: 1.0, 1: 0.4, 2: 0.15, 3: 0.05}
LOD_STEP = {0: 1.0, 1: 1.6, 2: 2.8, 3: 5.0}


class ReliefBuilder(Builder):
    """A figure or prop drawn from one side: the picture itself is the front, a rounded relief gives it real volume
    (depth follows the outline: thin parts become round, wide parts a plump lens) and the back is a softened mirror."""
    kind = "character"

    def __init__(self, subject, cam, opts):
        super().__init__(subject, cam, opts)
        m = subject.metrics
        pad = 2
        W, H = subject.size
        self.crop = (max(m["x0"] - pad, 0), max(m["y0"] - pad, 0), min(m["x1"] + pad, W), min(m["y1"] + pad, H))
        cw, chh = self.crop[2] - self.crop[0], self.crop[3] - self.crop[1]
        preset = dict(RELIEF_PRESETS.get(subject.kind, RELIEF_PRESETS["character"]))
        for k in ("rcap", "roundness", "lens", "smooth", "bend", "back_ratio"):
            if opts.get(k) is not None:
                preset[k] = float(opts[k])
        self.spec = ReliefSpec(**preset)
        self.longest = float(max(m["width"], m["height"]))
        self.front = self._chart(cw, chh, "front", crop=self.crop, density=1.0)
        self.back = self._chart(cw, chh, "back", crop=self.crop, density=float(opts.get("back_density", 0.5)))
        self.max_faces = int(opts.get("max_faces", 40000))
        self.info.update(relief=dict(preset), crop=list(self.crop))
        self.notes.append("relief: the picture is the front; depth is inflated from the outline (a rolling ball of at most "
                          f"{preset['rcap'] * 100:.1f}% of the size), so the sides are real geometry but their exact "
                          "shape is an educated guess")
        self.notes.append("relief: the back is a softened mirror of the front; pass a back image to replace it")

    def layers(self, lod=0, max_faces=0):
        lod = min(lod, 3)
        budget = int((max_faces or self.max_faces) * LOD_FACE_FRACTION[lod]) if (max_faces or self.max_faces) else 0
        L = build_relief_layer(self.s.alpha, self.cam, self.spec, self.front, self.back, self.crop, self.longest,
                               role="body", name="body", lod=LOD_STEP[lod], max_faces=max(budget, 800))
        return [L]


# ------------------------------------------------------------------------------------------------ island
def _row_extents(alpha: np.ndarray):
    hard = alpha >= 0.5
    H, W = hard.shape
    has = hard.any(1)
    xl = np.where(has, np.argmax(hard, 1), np.nan).astype(np.float64)
    xr = np.where(has, W - np.argmax(hard[:, ::-1], 1), np.nan).astype(np.float64)
    return has, xl, xr


def _smooth_rows(v: np.ndarray, med: int, sig: float) -> np.ndarray:
    from scipy import ndimage as ndi
    ok = np.isfinite(v)
    if not ok.any():
        return v
    idx = np.arange(len(v))
    f = np.interp(idx, idx[ok], v[ok])
    f = ndi.median_filter(f, size=med, mode="nearest")
    return ndi.gaussian_filter1d(f, sig, mode="nearest")


class IslandBuilder(Builder):
    """A floating island (or any 'plateau on a rock' prop) seen from above at an angle: a diamond-ish plateau, a rock
    cone below it whose left / right outline is read from the picture, and everything that rises above the plateau
    (castle, spires, trees) as relief patches standing on it."""
    kind = "island"
    tier = "B"

    def collision_groups(self, layers):
        from .collision import kmeans_hulls
        body = np.concatenate([L.verts for L in layers if L.name in ("island_rock", "island_top")])
        groups = [body]                                                # rock + plateau: convex enough for one hull
        card = [L.verts for L in layers if L.role == "skin"]
        if card:
            groups += kmeans_hulls(np.concatenate(card), int(self.opts.get("collision_hulls", 4)))
        return groups

    @staticmethod
    def camera(subject: Subject, opts: Dict[str, object]) -> OrthoCamera:
        W, H = subject.size
        return OrthoCamera(W, H, float(opts.get("elev") or 28.0))

    def __init__(self, subject, cam, opts):
        super().__init__(subject, cam, opts)
        s = subject
        has, xl, xr = _row_extents(s.alpha)
        ys = np.nonzero(has)[0]
        y_top, y_bot = int(ys.min()), int(ys.max())
        xl_s, xr_s = _smooth_rows(xl, 31, 5.0), _smooth_rows(xr, 31, 5.0)
        width = np.where(has, xr_s - xl_s, 0.0)
        lo = y_top + int(0.25 * (y_bot - y_top))
        y_w = int(lo + np.argmax(width[lo:y_bot]))                  # widest row below the castle: the plateau's side corners
        self.y_w = y_w
        xc = 0.5 * (xl_s[y_w] + xr_s[y_w])
        D = 0.5 * (xr_s[y_w] - xl_s[y_w])
        self.p_exp = float(opts.get("plateau_p", 1.8))
        e = cam.e
        K0 = 28
        yk = np.linspace(y_w, y_bot, K0)
        # rock outline below the corners: monotone, shrinking to the tip
        rl = np.minimum.accumulate(np.clip(xc - np.interp(yk, np.arange(len(xl_s)), xl_s), 0.0, None))
        rr = np.minimum.accumulate(np.clip(np.interp(yk, np.arange(len(xr_s)), xr_s) - xc, 0.0, None))
        rl[-1] = rr[-1] = 0.0
        rl[0], rr[0] = xc - xl_s[y_w], xr_s[y_w] - xc
        hk = -(yk - y_w) / np.cos(e)                               # height (negative = below the plateau)
        self.rho_l, self.rho_r, self.h_k = rl[::-1].copy(), rr[::-1].copy(), hk[::-1].copy()   # bottom (tip) -> top (rim)
        base = cam.ground_point(xc, float(y_w), 0.0)
        self.centre = (float(base[0]), float(base[2]))
        self.D = float(D)
        prof = np.stack([0.5 * (self.rho_l + self.rho_r), self.h_k], 1)
        plen = float(np.hypot(np.diff(prof[:, 0]), np.diff(prof[:, 1])).sum())
        dens = float(opts.get("density", 1.0))
        half = int(round(np.pi * D))
        self.rock_chart = self._chart(half, int(np.ceil(plen)) + 1, "param", fill="fold2", density=dens, params={"wrap_u": True})
        self.top_chart = self._chart(int(round(2 * D)), int(round(2 * D)), "param", fill="planar180",
                                     density=float(opts.get("top_density", dens)))
        self.info.update(elev=float(cam.elev), plateau_half_diagonal_px=self.D, rock_depth_px=float(-self.h_k.min()),
                         plateau_row=y_w)
        self.notes.append(f"island: camera elevation {cam.elev:.0f} deg (assumed; pass elev= to change), plateau half-diagonal "
                          f"{D:.0f} px, rock {(-self.h_k.min()):.0f} px deep - the outline of the rock is read from the picture, "
                          "its far side is a turned copy of the near side")
        self.notes.append("island: buildings inside the plateau outline are painted on the plateau; what rises above its "
                          "back edge (spires) stands up as thin relief")

    def _g(self, theta: np.ndarray) -> np.ndarray:
        p = self.p_exp
        s, c = np.abs(np.sin(theta)), np.abs(np.cos(theta))
        return (s ** p + c ** p) ** (-1.0 / p)

    def _body_layers(self, lod: int) -> List[Layer]:
        lod = min(lod, 3)
        nth = {0: 128, 1: 64, 2: 32, 3: 16}[lod]
        kk = {0: 28, 1: 16, 2: 9, 3: 5}[lod]
        sel = np.unique(np.round(np.linspace(0, len(self.h_k) - 1, kk)).astype(int))
        rl, rr, hk = self.rho_l[sel], self.rho_r[sel], self.h_k[sel]
        th = np.arange(nth) * (2 * np.pi / nth) - np.pi
        w = 0.5 * (1.0 + np.sin(th))
        g = self._g(th)
        rho = (rl[:, None] * (1 - w)[None, :] + rr[:, None] * w[None, :]) * g[None, :]
        prof = np.stack([0.5 * (rl + rr), hk], 1)
        c = self.centre
        rock = revolve(prof, nth, c, self.rock_chart, name="island_rock", fold=True, rho=rho)
        top = disc((c[0], 0.0, c[1]), float(rho[-1].max()), nth, {0: 16, 1: 10, 2: 5, 3: 2}[lod], self.top_chart, up=True,
                   name="island_top", radial=rho[-1] / max(rho[-1].max(), 1e-9))
        return [trim_to_matte(snap_to_matte(L, self.cam, self.s.alpha), self.cam, self.s.alpha, margin=8.0) for L in (rock, top)]

    def layers(self, lod=0, max_faces=0):
        body = self._body_layers(lod)
        if "_skin_ctx" not in self.__dict__:
            self._castle(self._body_layers(0))
            self.__dict__.setdefault("_skin_ctx", None)
        sk = skin_build(self, lod)
        return body + ([sk] if sk is not None else [])

    def _castle(self, body: List[Layer]) -> Optional[Layer]:
        """Everything above the plateau's centre row stands on an upright card through the plateau centre (the left and
        right corners of the plateau lie in that plane too); it is inflated like a figure, so towers become round."""
        H, W = self.s.alpha.shape
        rows = np.broadcast_to(np.arange(H)[:, None], (H, W))
        above = rows < self.y_w
        t_c = self.cam.project(np.array([[self.centre[0], 0.0, self.centre[1]]]))[2][0]
        card_t = (t_c + np.tan(self.cam.e) * (self.y_w - rows)).astype(np.float32)
        spec = ReliefSpec(rcap=float(self.opts.get("card_rcap", 0.09)), roundness=0.9, lens=0.5, smooth=1.5)
        L = skin_layer(self, body, min_area=40, extra=above, extra_base=card_t, spec=spec, name="castle")
        if L is not None:
            self.notes.append("island: everything above the plateau's centre row is an upright relief card through the "
                              "plateau centre (the castle is a plump plate, not a modelled building)")
        return L
