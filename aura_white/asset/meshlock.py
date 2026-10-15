"""Mesh-lock: take a mesh somebody else generated (Meshy, Tripo, Hunyuan3D, TRELLIS, a sculpt ...) and *lock* it to the
picture.

The reference view is exact because the picture is projected through an orthographic camera fitted to the mesh; what the
picture does not show (the sides and the back) comes from the mesh's own shape, and its colours are solved on the
surface itself (see ``meshfill``) instead of being smeared sideways or mirrored.

    fit_pose        find the orthographic camera / pose that makes the mesh's outline match the picture's matte
    lock_outline    slide the few vertices that stick out of the matte back inside, in the picture plane
    residual_skin   add thin relief patches for picture pixels the mesh does not cover
    unwrap          xatlas UVs (optional dependency) with a guarded fallback
    MeshLockBuilder the builder the asset pipeline drives (LOD chain, one shared atlas, collision hulls)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import meshio
from .camera import OrthoCamera, rasterize_ortho, render_view
from .layers import Chart, Layer


# ------------------------------------------------------------------------------------------------ pose
def _rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass
class Pose:
    centre: np.ndarray           # point of the source mesh that is moved to the origin
    yaw: float                   # degrees about +Y (positive turns the front towards +X)
    elev: float                  # camera elevation, degrees
    scale: float                 # source mesh units -> picture pixels
    dx: float                    # picture-pixel shift of the origin: x
    dy: float                    # world Y shift (pixels)
    iou: float = 0.0
    note: str = ""

    def apply(self, v: np.ndarray) -> np.ndarray:
        P = (np.asarray(v, np.float64) - self.centre) @ _rot_y(self.yaw).T * self.scale
        return P + np.array([self.dx, self.dy, 0.0])


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


class _Fitter:
    """Silhouette matcher.  The cost is a *soft* IoU of Gaussian-blurred masks, so thin parts (a scythe's shaft, a
    ribbon) still pull the pose in from a few pixels away; the blur is tightened in stages."""

    def __init__(self, alpha: np.ndarray, v: np.ndarray, f: np.ndarray, long_side: int = 240):
        H, W = alpha.shape
        self.W, self.H = W, H
        self.k = long_side / max(W, H)
        self.h, self.w = max(8, int(round(H * self.k))), max(8, int(round(W * self.k)))
        from PIL import Image
        m = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)).resize((self.w, self.h), Image.Resampling.BOX)
        self.matte_f = np.asarray(m).astype(np.float32) / 255.0
        self.matte = self.matte_f >= 0.5
        self.area = float(self.matte_f.sum())
        ys, xs = np.nonzero(self.matte)
        self.mcx, self.mcy = (xs.mean() + 0.5) / self.k, (ys.mean() + 0.5) / self.k
        self.v, self.f = v, f
        self.centre = 0.5 * (v.min(0) + v.max(0))
        self.v0 = v - self.centre
        self.diag = float(np.linalg.norm(v.max(0) - v.min(0)))
        self._blur: Dict[float, np.ndarray] = {}

    def matte_blur(self, sigma: float) -> np.ndarray:
        if sigma not in self._blur:
            from scipy import ndimage as ndi
            self._blur[sigma] = ndi.gaussian_filter(self.matte_f, sigma) if sigma > 0 else self.matte_f
        return self._blur[sigma]

    def raster(self, yaw, elev, s, dx, dy):
        cam = OrthoCamera(self.W, self.H, elev)
        P = self.v0 @ _rot_y(yaw).T * s + np.array([dx, dy, 0.0])
        sx, sy, t = cam.project(P)
        tid, _, _ = rasterize_ortho(np.stack([sx * self.k, sy * self.k], 1), t, self.f, self.h, self.w, nearest_is_larger=True)
        return tid >= 0

    def init_scale_shift(self, yaw, elev):
        """Closed-form start: area ratio for the scale, centroid for the shift."""
        s = 0.8 * max(self.W, self.H) / max(self.diag, 1e-9)
        m = self.raster(yaw, elev, s, 0.0, 0.0)
        a = float(m.sum())
        if a < 4:
            return s, 0.0, 0.0
        s2 = s * np.sqrt(self.area / a)
        m = self.raster(yaw, elev, s2, 0.0, 0.0)
        if m.sum() < 4:
            return s2, 0.0, 0.0
        ys, xs = np.nonzero(m)
        cx, cy = (xs.mean() + 0.5) / self.k, (ys.mean() + 0.5) / self.k
        dx = self.mcx - cx
        dy = -(self.mcy - cy) / max(np.cos(np.radians(elev)), 0.2)     # picture rows grow downwards, world Y upwards
        return s2, dx, dy

    def cost(self, p, yaw0, sigma=0.0, lam=0.004):
        yaw, elev, ls, dx, dy = p
        if not (-1.0 <= elev <= 70.0):
            return 2.0 + abs(elev) * 0.01
        m = self.raster(yaw, elev, float(np.exp(ls)), dx, dy)
        if sigma > 0:
            from scipy import ndimage as ndi
            a, b = ndi.gaussian_filter(m.astype(np.float32), sigma), self.matte_blur(sigma)
            iou = float(np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-9))
        else:
            iou = _mask_iou(m, self.matte)
        return 1.0 - iou + lam * abs(yaw - yaw0) / 10.0


def fit_pose(alpha: np.ndarray, verts: np.ndarray, faces: np.ndarray, elev: Optional[float] = None,
             yaw: Optional[float] = None, yaws=(0.0, 90.0, 180.0, 270.0), elevs=(0.0, 12.0, 25.0, 40.0, 55.0),
             refine: bool = True, prefer_front: float = 0.04, yaw_pull: float = 0.01) -> Pose:
    """Camera and pose of the mesh that best reproduce the picture's silhouette (maximum IoU).  The mesh is assumed to
    stand on +Y.  Generators put the front on +Z, so that reading wins unless another quarter turn matches the outline
    clearly better (`prefer_front`, in IoU); a thin prop seen mirrored (a scythe) does switch.  `elev` fixes the camera
    elevation when the artwork's angle is known; `yaw` fixes the turn."""
    from scipy.optimize import minimize
    fv, ff = (verts, faces) if len(faces) <= 4000 else meshio.simplify(verts, faces, 4000)
    fit = _Fitter(alpha, fv, ff, long_side=200)
    base_yaws = (float(yaw),) if yaw is not None else tuple(yaws)
    el_list = (float(elev),) if elev is not None else tuple(elevs)
    sig0 = max(2.0, 0.012 * max(fit.w, fit.h))

    def full(p):
        return np.array([p[0], elev if elev is not None else p[1], p[2], p[3], p[4]], np.float64)

    def run(x0, y0, sigma, maxfev):
        step = np.diag([5.0 if sigma > 1 else 2.0, 0.0 if elev is not None else (4.0 if sigma > 1 else 1.5),
                        0.04 if sigma > 1 else 0.012, 0.02 * fit.W if sigma > 1 else 0.006 * fit.W,
                        0.02 * fit.H if sigma > 1 else 0.006 * fit.H])
        sim = np.array([x0] + [x0 + d for d in step])
        r = minimize(lambda p: fit.cost(full(p), y0, sigma, lam=yaw_pull), x0, method="Nelder-Mead",
                     options=dict(initial_simplex=sim, maxfev=maxfev, xatol=0.02, fatol=1e-4))
        return r.x, float(r.fun)

    start = {}
    for y0 in base_yaws:
        cands = []
        for e0 in el_list:
            s0, dx, dy = fit.init_scale_shift(y0, e0)
            cands.append((fit.cost((y0, e0, np.log(s0), dx, dy), y0, sig0, lam=yaw_pull), e0, s0, dx, dy))
        c0, e0, s0, dx, dy = min(cands)
        start[y0] = (c0, np.array([y0, e0, np.log(s0), dx, dy], np.float64))
    # only the most promising quarter turns (and the +Z reading) are worth a full search
    order = sorted(start, key=lambda k: start[k][0])
    todo = order[:2] + ([0.0] if 0.0 in start and 0.0 not in order[:2] else [])
    fam = {y0: start[y0] for y0 in start}
    for y0 in todo:
        x, f_ = start[y0][1], start[y0][0]
        if refine:
            x, f_ = run(x, y0, sig0, 110)
        fam[y0] = (f_, x)
    best_y = min(fam, key=lambda k: fam[k][0])
    if yaw is None and 0.0 in fam and fam[0.0][0] <= fam[best_y][0] + prefer_front:
        best_y = 0.0
    _, x = fam[best_y]
    if refine:
        x, _ = run(x, best_y, 0.5 * sig0, 90)
        x, _ = run(x, best_y, 0.0, 70)
    x = full(x)
    pose = Pose(fit.centre.copy(), float(x[0]), float(x[1]), float(np.exp(x[2])), float(x[3]), float(x[4]))
    fine = _Fitter(alpha, fv, ff, long_side=640)
    pose.iou = _mask_iou(fine.raster(pose.yaw, pose.elev, pose.scale, pose.dx, pose.dy), fine.matte)
    return pose


# ------------------------------------------------------------------------------------------------ outline lock
def _compact_vertices(v: np.ndarray, f: np.ndarray):
    """Drop unreferenced vertices without touching the order of the faces."""
    used = np.zeros(len(v), bool)
    used[f.reshape(-1)] = True
    remap = np.cumsum(used) - 1
    return v[used], remap[f]


def lock_outline(v: np.ndarray, f: np.ndarray, cam: OrthoCamera, alpha: np.ndarray, inset: float = 0.5,
                 margin: float = 3.0):
    """Make the mesh's outline in the reference view fit inside the matte.  A face with a vertex more than `margin` px
    outside is dropped (dragging such a vertex all the way in would stretch the face into a long sliver); every other
    vertex outside the matte slides to the nearest matte pixel *in the picture plane* (its depth along the view ray is
    kept), so the picture's pixel stays on the same surface point.
    -> (verts, faces, info) - ``info['keep']`` flags the input faces that survive, in order."""
    from scipy import ndimage as ndi
    hard = alpha >= 0.5
    H, W = hard.shape
    px, py, t = cam.project(v)
    d_out, (ny, nx) = ndi.distance_transform_edt(~hard, return_indices=True)
    ix = np.clip(np.floor(px).astype(np.int64), 0, W - 1)
    iy = np.clip(np.floor(py).astype(np.int64), 0, H - 1)
    frame = (px < 0) | (px > W) | (py < 0) | (py > H)
    dist = np.where(frame, 1e9, d_out[iy, ix])
    far = dist > margin
    keep = ~far[f].any(1)
    f2 = f[keep]
    outside = ((~hard[iy, ix]) | frame) & ~far
    used = np.zeros(len(v), bool)
    used[f2.reshape(-1)] = True
    outside &= used
    v2 = v.copy()
    if outside.any():
        tx = nx[iy[outside], ix[outside]] + 0.5
        ty = ny[iy[outside], ix[outside]] + 0.5
        dx, dy = tx - px[outside], ty - py[outside]
        nrm = np.maximum(np.hypot(dx, dy), 1e-9)
        v2[outside] = cam.unproject(tx + inset * dx / nrm, ty + inset * dy / nrm, t[outside])
    v3, f3 = _compact_vertices(v2, f2)
    return v3, f3, dict(snapped_vertices=int(outside.sum()), trimmed_faces=int((~keep).sum()), keep=keep)


def grow_to_matte(v: np.ndarray, f: np.ndarray, cam: OrthoCamera, alpha: np.ndarray, max_move: float = 10.0,
                  iters: int = 4, min_inside: float = 1.2):
    """The opposite of `lock_outline`: where the mesh's outline lies *inside* the matte (a generated shaft or ribbon is
    a little thinner than the artwork's), slide the outline vertices outwards, in the picture plane, onto the matte's
    edge.  Only vertices on the outer outline move, and never by more than `max_move` px.  -> verts, moved count"""
    from scipy import ndimage as ndi
    hard = alpha >= 0.5
    H, W = hard.shape
    sd = ndi.distance_transform_edt(hard)
    gy, gx = np.gradient(ndi.gaussian_filter(sd, 1.0))
    v = v.copy()
    moved = np.zeros(len(v), bool)
    for _ in range(iters):
        px, py, t = cam.project(v)
        sx = np.stack([px, py], 1)
        tid, _, dep = rasterize_ortho(sx, t, f, H, W, nearest_is_larger=True)
        M = tid >= 0
        edge = M & ~ndi.binary_erosion(M, iterations=2)
        ix = np.clip(np.floor(px).astype(np.int64), 0, W - 1)
        iy = np.clip(np.floor(py).astype(np.int64), 0, H - 1)
        z = dep[iy, ix]
        d = sd[iy, ix]
        sel = edge[iy, ix] & np.isfinite(z) & (t >= z - 2.5) & (d > min_inside) & (d <= max_move) & hard[iy, ix]
        if not sel.any():
            break
        g = np.stack([gx[iy[sel], ix[sel]], gy[iy[sel], ix[sel]]], 1)
        n = np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
        step = np.minimum(d[sel] - 0.5, 0.75 * max_move)[:, None]
        p2 = np.stack([px[sel], py[sel]], 1) - g / n * step
        v[sel] = cam.unproject(p2[:, 0], p2[:, 1], t[sel])
        moved |= sel
    return v, int(moved.sum())


def orient_outward(v: np.ndarray, f: np.ndarray, cam: OrthoCamera):
    """Flip the winding when most of what the reference camera sees faces away from it (an inside-out source mesh)."""
    sx, sy, t = cam.project(v)
    tid, _, _ = rasterize_ortho(np.stack([sx * 0.25, sy * 0.25], 1), t, f, max(8, cam.height // 4), max(8, cam.width // 4),
                                nearest_is_larger=True)
    hit = tid >= 0
    if not hit.any():
        return f, False
    tri = f[tid[hit]]
    n = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    facing = n @ cam.to_camera
    if (facing < 0).mean() > 0.5:
        return f[:, [0, 2, 1]], True
    return f, False


def uplift_mask(s, cam: OrthoCamera, v: np.ndarray, f: np.ndarray):
    """Towers and spires the generated mesh left too short.  Where a structure in the picture rises above the top edge of a
    wall, the mesh shows the *ground far behind* at those pixels, and the artwork's tower colours would be painted onto it
    (long dark streaks across the floor when the model turns).  Such pixels are found as busy (high texture energy) regions on
    ground-facing mesh surface that touch a wall rising in front of them.  -> (mask, wall depth field) or (None, None)."""
    from scipy import ndimage as ndi
    tid, bary, t = render_view(cam, v, f)
    hit = tid >= 0
    if not hit.any():
        return None, None
    tri = f[np.maximum(tid, 0)]
    n = np.cross(v[tri[..., 1]] - v[tri[..., 0]], v[tri[..., 2]] - v[tri[..., 0]])
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    up = n[..., 1]
    ground = hit & (up > 0.8)
    wall = hit & (np.abs(up) < 0.45)
    if ground.sum() < 0.04 * hit.sum() or wall.sum() < 0.02 * hit.sum():
        return None, None
    core = ndi.binary_erosion(ground, iterations=25)
    if core.sum() < 2000:
        return None, None
    lum = (0.2126 * s.rgb[..., 0] + 0.7152 * s.rgb[..., 1] + 0.0722 * s.rgb[..., 2]).astype(np.float32)
    E = ndi.uniform_filter(ndi.gaussian_gradient_magnitude(lum, 1.5), 13)
    base_e = float(np.median(E[core]))
    cand = ndi.binary_opening(ground & (E > 3.0 * base_e), structure=np.ones((5, 5)))
    lab, k = ndi.label(ndi.binary_closing(cand, structure=np.ones((5, 5)), iterations=2))
    touch = ndi.binary_dilation(wall, iterations=10)
    # only the part of the ground nearer the camera than the arena's centre can be hidden behind a wall that rises in front
    rows = np.nonzero(ground.any(1))[0]
    y_front = rows.min() + 0.45 * (rows.max() - rows.min())
    keep = np.zeros(k + 1, bool)
    for i in range(1, k + 1):
        comp = lab == i
        yy = np.nonzero(comp.any(1))[0]
        if comp.sum() > 400 and (comp & touch).any() and yy.mean() > y_front:
            keep[i] = True
    mask = keep[lab]
    if mask.sum() < 300:
        return None, None
    mask = ndi.binary_dilation(mask, iterations=3) & (s.alpha >= 0.5)
    from .proxies import skin_base_depth
    base = skin_base_depth(np.where(wall, t, 0.0), wall, cam)
    return mask, base


def residual_skin(s, cam: OrthoCamera, v: np.ndarray, f: np.ndarray, spec=None, max_faces: int = 6000,
                  min_area: int = 24, min_width: float = 3.5, uplift: bool = False):
    """Relief patches for the picture pixels the mesh does not cover (a ribbon end the generator dropped, a hair strand) and
    for towers it left too short (`uplift_mask`).  -> (verts, faces, pixel count, uplift pixel count) or None."""
    from scipy import ndimage as ndi
    from .proxies import nearest_fill
    from .relief import ReliefSpec, build_relief_layer
    _, _, t = render_view(cam, v, f)
    covered = np.isfinite(t)
    hard = s.alpha >= 0.5
    resid = hard & ~ndi.binary_dilation(covered, iterations=2)
    resid = ndi.binary_opening(resid, structure=np.ones((3, 3)), iterations=2)       # a 1-3 px rim is not a missing part
    lab, n = ndi.label(resid, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    if n:
        area = np.bincount(lab.ravel())
        keep = area >= min_area
        keep[0] = False
        # hair-thin leftovers (outline wobble) are not worth geometry: a patch needs a body at least ~7 px across somewhere
        width = ndi.maximum(ndi.distance_transform_edt(resid), lab, index=np.arange(n + 1))
        keep &= np.asarray(width) >= min_width
    resid = keep[lab]
    up_mask, up_base = uplift_mask(s, cam, v, f) if uplift else (None, None)
    if not resid.any() and up_mask is None:
        return None
    rows = np.arange(t.shape[0], dtype=np.float32)[:, None]
    tan_e = float(np.tan(cam.e))
    # a rim patch continues the surface right next to it, as one flat card per piece: a per-pixel depth would jump between
    # the neighbouring surfaces and stretch thin walls across the whole depth of the model
    near = nearest_fill(np.where(covered, t, 0.0).astype(np.float32), covered)
    base = near.copy()
    lab2, n2 = ndi.label(resid, structure=np.ones((3, 3)))
    if n2:
        med = np.asarray(ndi.median(near, lab2, index=np.arange(1, n2 + 1)))
        base[lab2 > 0] = med[lab2[lab2 > 0] - 1]
    up_px = 0
    if up_mask is not None:
        # a tower stands upright on the wall below it: one vertical plane per tower
        b0 = (up_base + tan_e * rows).astype(np.float32)
        labu, nu = ndi.label(up_mask, structure=np.ones((3, 3)))
        medu = np.asarray(ndi.median(b0, labu, index=np.arange(1, nu + 1)))
        plane = np.zeros_like(b0)
        plane[labu > 0] = medu[labu[labu > 0] - 1]
        base = np.where(up_mask, plane - tan_e * rows, base)
        resid = resid | up_mask
        up_px = int(up_mask.sum())
    ys, xs = np.nonzero(resid)
    pad = 3
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + 1 + pad, s.size[0])
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + 1 + pad, s.size[1])
    longest = max(s.metrics.get("width", s.size[0]), s.metrics.get("height", s.size[1]))
    spec = spec or ReliefSpec(rcap=0.006, roundness=0.7, smooth=0.8)
    alpha = np.where(resid, np.maximum(s.alpha, 0.5), 0.0).astype(np.float32)
    ch = Chart(-1, x1 - x0, y1 - y0, "front", crop=(x0, y0, x1, y1))
    L = build_relief_layer(alpha, cam, spec, ch, None, (x0, y0, x1, y1), longest, role="skin", name="skin", lod=1.0,
                           base_t=base, max_faces=max_faces)
    if L is None:
        return None
    return L.verts, L.faces, int(resid.sum()), up_px


# ------------------------------------------------------------------------------------------------ UVs
def _laplacian_smooth(v: np.ndarray, f: np.ndarray, iters: int = 30, lam: float = 0.5, mu: float = -0.53):
    """Taubin smoothing (no shrinkage): charts are cut on a calmer copy of the surface, so a noisy generated mesh does not
    shatter into thousands of tiny islands."""
    from scipy import sparse
    n = len(v)
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    d = np.maximum(np.asarray(A.sum(1)).ravel(), 1.0)
    L = sparse.diags(1.0 / d) @ A
    x = v.astype(np.float64).copy()
    for _ in range(iters):
        x = x + lam * (L @ x - x)
        x = x + mu * (L @ x - x)
    return x


def _box_unwrap(v: np.ndarray, f: np.ndarray, size: int = 2048):
    """Fallback without xatlas: faces grouped by dominant axis, planar projection, shelf packing.  Surfaces that lie behind
    one another overlap in the atlas - fine for a quick look, install xatlas for real UVs."""
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area = 0.5 * np.linalg.norm(n, axis=1)
    ax = np.argmax(np.abs(n), axis=1)
    sg = (np.take_along_axis(n, ax[:, None], 1)[:, 0] < 0).astype(int)
    grp = ax * 2 + sg
    axes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    s = np.sqrt(0.5 * size * size / max(float((area * np.abs(np.take_along_axis(n, ax[:, None], 1)[:, 0]) /
                                                  np.maximum(2 * area, 1e-12)).sum()), 1e-12))
    for _ in range(12):
        uv = np.zeros((len(f), 3, 2))
        x = y = rowh = 0.0
        pad = 6.0
        ok = True
        for g in range(6):
            sel = np.flatnonzero(grp == g)
            if sel.size == 0:
                continue
            a0, a1 = axes[g // 2]
            p = np.stack([v[f[sel]][..., a0], v[f[sel]][..., a1]], -1) * s
            if g % 2:
                p[..., 0] *= -1
            lo, hi = p.reshape(-1, 2).min(0), p.reshape(-1, 2).max(0)
            wdt, hgt = hi - lo + 2 * pad
            if wdt > size:
                ok = False
                break
            if x + wdt > size:
                x, y, rowh = 0.0, y + rowh, 0.0
            uv[sel] = p - lo + np.array([x + pad, y + pad])
            x += wdt
            rowh = max(rowh, hgt)
        if ok and y + rowh <= size:
            return uv.astype(np.float32), np.asarray(grp, np.int32), size
        s *= 0.85
    raise RuntimeError("box unwrap could not fit the atlas")


def unwrap(v: np.ndarray, f: np.ndarray, smooth: int = 30):
    """-> (corner_uv (F,3,2) in texels of an (aw, ah) atlas, chart id per face, (aw, ah), how)."""
    try:
        import xatlas
    except ImportError:
        uv, grp, size = _box_unwrap(v, f)
        return uv, grp, (size, size), "box projection (install `xatlas` for real UVs)"
    vs = _laplacian_smooth(v, f, smooth) if smooth else v
    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(vs, np.float32), np.asarray(f, np.uint32))
    po = xatlas.PackOptions()
    po.padding = 4
    po.bilinear = True
    atlas.generate(xatlas.ChartOptions(), po)
    _, idx, uv = atlas[0]
    aw, ah = int(atlas.width), int(atlas.height)
    idx = np.asarray(idx, np.int64)
    cuv = (uv[idx] * np.array([aw, ah], np.float64)).astype(np.float32)
    n, lab = meshio.components(idx, len(uv))
    return cuv, lab.astype(np.int32), (aw, ah), f"xatlas ({n} charts)"


def transfer_uv(src_v: np.ndarray, src_f: np.ndarray, src_cuv: np.ndarray, src_chart: np.ndarray, dst_v: np.ndarray,
                dst_f: np.ndarray, n_samples: int = 500_000, seed: int = 0):
    """UVs for a coarser mesh that lies on the same surface (a LOD): every corner takes the UV of the nearest surface point
    of the finer mesh *within the chart the face belongs to*, so one atlas serves the whole LOD chain."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    tri = src_v[src_f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    p = area / max(area.sum(), 1e-12)
    fi = rng.choice(len(src_f), size=n_samples, p=p)
    b = rng.dirichlet((1.0, 1.0, 1.0), size=n_samples)
    pos = (tri[fi] * b[..., None]).sum(1)
    uv = (src_cuv[fi].astype(np.float64) * b[..., None]).sum(1)
    cid = src_chart[fi].astype(np.float64)
    diag = float(np.linalg.norm(src_v.max(0) - src_v.min(0)))
    big = 20.0 * diag
    tree3 = cKDTree(pos)
    tree4 = cKDTree(np.c_[pos, cid * big])
    cen = dst_v[dst_f].mean(1)
    _, j = tree3.query(cen, workers=-1)
    fc = cid[j]
    corners = dst_v[dst_f]                                           # (F,3,3)
    q = np.concatenate([corners.reshape(-1, 3), np.repeat(fc, 3)[:, None] * big], 1)
    _, jj = tree4.query(q, workers=-1)
    out = uv[jj].reshape(-1, 3, 2)
    # a face whose corners landed far apart in the chart (it straddles a fold) collapses to its centroid's UV
    e3 = np.linalg.norm(corners[:, [1, 2, 0]] - corners, axis=2).max(1)
    e2 = np.linalg.norm(out[:, [1, 2, 0]] - out, axis=2).max(1)
    s_src = 0.5 * np.linalg.norm(src_cuv[:, [1, 2, 0]] - src_cuv, axis=2).max(1)
    ratio = float(np.median(np.linalg.norm(src_cuv[:, 1] - src_cuv[:, 0], axis=1) /
                            np.maximum(np.linalg.norm(src_v[src_f[:, 1]] - src_v[src_f[:, 0]], axis=1), 1e-9)))
    bad = e2 > 4.0 * ratio * e3 + 8.0
    if bad.any():
        out[bad] = uv[j][bad][:, None, :]
    return out.astype(np.float32), int(bad.sum()), fc.astype(np.int64)


def front_faces(v: np.ndarray, f: np.ndarray, cam: OrthoCamera, min_facing: float = 0.12, min_visible: float = 0.6) -> np.ndarray:
    """Faces the reference camera really shows (mostly unoccluded, not edge-on).  They are textured by *planar projection of
    the picture itself*, so the reference view is reproduced texel for pixel; everything else goes through the surface
    unwrap.  A face smaller than a pixel counts as shown when it owns at least one pixel."""
    from .camera import render_view as _rv
    tid, _, _ = _rv(cam, v, f)
    cnt = np.bincount(tid[tid >= 0], minlength=len(f)).astype(np.float64)
    px, py, _ = cam.project(v)
    a = 0.5 * np.abs((px[f[:, 1]] - px[f[:, 0]]) * (py[f[:, 2]] - py[f[:, 0]]) - (px[f[:, 2]] - px[f[:, 0]]) * (py[f[:, 1]] - py[f[:, 0]]))
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    facing = n @ cam.to_camera
    shown = np.where(a < 2.0, cnt >= 1.0, cnt >= min_visible * a)
    return shown & (facing >= min_facing)


def front_layout(cw: int, chh: int, aw: int, ah: int, cap: int, pad: int = 8, min_mesh: float = 0.4, min_front: float = 0.5):
    """Densities of the picture's front chart (crop cw x chh px) and of the unwrapped rest (native aw x ah) so that both fit a
    cap x cap atlas as a shelf packer places them (front first, the rest beside it or below it).  The front chart keeps
    1:1 whenever it can - that is what makes the reference view exact - and the unseen sides give way first."""
    best = None
    for k in range(0, 11):
        sf = 1.0 - 0.05 * k
        if sf < min_front:
            break
        wf, hf = int(round(cw * sf)) + 2 * pad, int(round(chh * sf)) + 2 * pad
        if wf > cap or hf > cap:
            continue
        side = min((cap - wf - 2 * pad) / aw, (cap - 2 * pad) / ah)          # the rest beside the front chart
        below = min((cap - 2 * pad) / aw, (cap - hf - 2 * pad) / ah)         # ... or below it
        m = max(side, below)
        if m >= min_mesh:
            return sf, float(min(m, 1.5))
        if best is None or m > best[1]:
            best = (sf, m)
    if best is None:                                                          # a huge crop: shrink it to what fits
        sf = max(0.2, min((cap - 2 * pad) / max(cw, chh), 1.0))
        return sf, 0.25
    return best[0], float(max(min(best[1], 1.5), 0.2))


# ------------------------------------------------------------------------------------------------ the builder
LOD_FRACTION = {0: 1.0, 1: 0.4, 2: 0.15, 3: 0.05}


def _chart_cuv(layer_faces, cuv):
    return cuv


class MeshLockBuilder:
    """Builder for the asset pipeline (same interface as ``proxies.Builder``) around a supplied mesh."""
    kind = "mesh"
    tier = "M"

    # -- camera: the pose fit has to happen first, so it lives here; the result is cached in the options dict
    @staticmethod
    def camera(subject, opts: Dict[str, object]) -> OrthoCamera:
        src = opts.get("proxy")
        if src is None:
            raise ValueError("mesh-lock needs proxy=<mesh file>")
        src_col = None
        if isinstance(src, (tuple, list)):
            v, f = np.asarray(src[0], np.float64), np.asarray(src[1], np.int64)
            if len(src) > 2 and src[2] is not None:
                src_col = (np.asarray(src[2][0], np.float64), np.asarray(src[2][1]))
            how = "arrays"
        elif opts.get("proxy_colour", True):
            v, f, src_col = meshio.load_mesh_coloured(src)
            how = str(src)
        else:
            v, f = meshio.load_mesh(src)
            how = str(src)
        opts["_src_col"] = src_col if opts.get("proxy_colour", True) else None
        n0 = len(f)
        v, f = meshio.weld(v, f, 1e-6 * float(np.linalg.norm(v.max(0) - v.min(0))))
        v, f = meshio.drop_specks(v, f)
        elev = opts.get("elev")
        pose = fit_pose(subject.alpha, v, f, elev=float(elev) if elev is not None else None,
                        yaw=float(opts["proxy_yaw"]) if opts.get("proxy_yaw") is not None else None)
        opts["_mesh"] = (v, f, n0, how)
        opts["_pose"] = pose
        W, H = subject.size
        return OrthoCamera(W, H, pose.elev)

    def __init__(self, subject, cam: OrthoCamera, opts: Dict[str, object]):
        from .relief import ReliefSpec
        self.s, self.cam, self.opts = subject, cam, dict(opts)
        self.charts: Dict[int, Chart] = {}
        self.notes: List[str] = []
        self.info: Dict[str, object] = {}
        v, f, n_src, how = opts["_mesh"]
        pose: Pose = opts["_pose"]
        self.pose = pose
        s = subject
        # 1. into the picture's world, outline locked, outward facing
        world = pose.apply(v)
        f, flipped = orient_outward(world, f, cam)
        hi_v, hi_f, lk = lock_outline(world, f, cam, s.alpha, margin=3.0)
        budget = int(opts.get("max_faces") or 40000)
        target = int(min(budget, max(2000, 0.9 * len(hi_f))))
        # 2. LOD0: quadric decimation of the locked mesh, locked again (decimation moves vertices a little)
        lo_v, lo_f = meshio.simplify(hi_v, hi_f, target)
        lo_v, grown = grow_to_matte(lo_v, lo_f, cam, s.alpha)
        lo_v, lo_f, lk0 = lock_outline(lo_v, lo_f, cam, s.alpha, margin=5.0)
        # 3. what the mesh does not cover of the picture stands up as thin relief
        skin = residual_skin(s, cam, lo_v, lo_f, ReliefSpec(rcap=float(opts.get("skin_rcap", 0.0015)), roundness=0.7, smooth=0.8),
                             max_faces=int(opts.get("skin_faces", 6000)), uplift=bool(opts.get("uplift", False)))
        skin_px = 0
        uplift_px = 0
        self.skin_faces = 0
        n_main = len(lo_f)
        if skin is not None:
            sv, sf, skin_px, uplift_px = skin
            self.skin_faces = len(sf)
            lo_f = np.concatenate([lo_f, sf + len(lo_v)])
            lo_v = np.concatenate([lo_v, sv])
            hi_f = np.concatenate([hi_f, sf + len(hi_v)])
            hi_v = np.concatenate([hi_v, sv])
        self.hi_mesh = (hi_v, hi_f)
        # 4. UVs for LOD0 (the part of the surface the picture shows head-on takes the picture itself as its texture; the
        #    rest is unwrapped), then every coarser LOD on the same atlas
        cap = int(opts.get("atlas_cap", 2048))
        want_front = bool(opts.get("front_chart", True))
        front = front_faces(lo_v, lo_f, cam) if want_front else np.zeros(len(lo_f), bool)
        if front.sum() < 50 or (~front).sum() < 50:
            front[:] = False
        ys, xs = np.nonzero(s.alpha >= 0.5)
        H_img, W_img = s.alpha.shape
        x0c, y0c = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
        x1c, y1c = min(W_img, int(xs.max()) + 3), min(H_img, int(ys.max()) + 3)
        cw, chh = x1c - x0c, y1c - y0c
        cuv = np.zeros((len(lo_f), 3, 2), np.float32)
        face_chart = np.zeros(len(lo_f), np.int32)
        chart_id = np.zeros(len(lo_f), np.int32)
        rest = ~front
        used = np.unique(lo_f[rest])
        remap = -np.ones(len(lo_v), np.int64)
        remap[used] = np.arange(len(used))
        sub_cuv, sub_lab, (aw, ah), how_uv = unwrap(lo_v[used], remap[lo_f[rest]], int(opts.get("uv_smooth", 30)))
        cuv[rest] = sub_cuv
        chart_id[rest] = sub_lab
        if front.any():
            px, py, _ = cam.project(lo_v)
            fu = np.stack([px - x0c, py - y0c], 1)
            cuv[front] = fu[lo_f[front]].astype(np.float32)
            face_chart[front] = 1
            front_id = int(sub_lab.max()) + 1 if rest.any() else 0
            chart_id[front] = front_id
        else:
            front_id = -1
        ch = Chart(0, aw, ah, "param", fill="mesh", density=1.0,
                   params=dict(style=str(opts.get("back", "soft")), grazing=float(opts.get("grazing", 0.22)),
                               detail=float(opts.get("detail", 0.0)), back_img=None, axis_x=float(pose.dx)))
        self.charts[0] = ch
        if front.any():
            s_f, s_m = front_layout(cw, chh, aw, ah, cap)
            self.charts[1] = Chart(1, cw, chh, "front", crop=(x0c, y0c, x1c, y1c), density=s_f)
            ch.density = s_m
        else:
            ch.density = float(np.clip((cap - 24) / max(aw, ah, 1), 0.25, 2.0))      # fill the atlas the pipeline will build
        sc = opts.get("_src_col")
        if sc is not None:
            from .meshfill import SourceColour
            ch.params["src_col"] = SourceColour(pose.apply(np.asarray(sc[0], np.float64)), sc[1])
            ch.params["skin_start"] = n_main
            self.info["source_colours"] = int(len(sc[0]))
        self._lods = [(lo_v, lo_f, cuv, face_chart)]
        n_lod = int(max(1, min(int(opts.get("lods", 4)), 4)))
        bad_total = 0
        for k in range(1, n_lod):
            tv, tf = meshio.simplify(lo_v, lo_f, max(200, int(len(lo_f) * LOD_FRACTION[k])))
            tuv, nbad, tfc = transfer_uv(lo_v, lo_f, cuv, chart_id, tv, tf)
            bad_total += nbad
            tv2, tf2, lkk = lock_outline(tv, tf, cam, s.alpha, margin=8.0)
            keep = lkk["keep"]
            self._lods.append((tv2, tf2, tuv[keep], (tfc[keep] == front_id).astype(np.int32) if front_id >= 0
                               else np.zeros(int(keep.sum()), np.int32)))
        self.info["front_chart_faces"] = int(front.sum())
        self.info.update(
            source_triangles=int(n_src), source=how, pose=dict(yaw=round(pose.yaw, 1), elev=round(pose.elev, 1),
                                                                  scale=round(pose.scale, 2)),
            silhouette_iou_fit=round(pose.iou, 4), snapped_vertices=lk["snapped_vertices"] + lk0["snapped_vertices"], grown_vertices=grown,
            trimmed_faces=lk["trimmed_faces"] + lk0["trimmed_faces"], skin_pixels=skin_px, uplift_pixels=uplift_px, unwrap=how_uv,
            lod_uv_fallback_faces=bad_total, flipped_winding=bool(flipped))
        self.notes.append(f"mesh-lock: {Path(str(how)).name if how != 'arrays' else 'supplied mesh'} "
                          f"({n_src:,} triangles) fitted to the picture (outline IoU {pose.iou:.3f}, camera elevation "
                          f"{pose.elev:.1f} deg, turned {pose.yaw:.0f} deg); its own shape is used for the sides and the back")
        if sc is not None:
            self.notes.append("mesh-lock: the supplied mesh carries its own colours; they colour the sides and the back "
                              "(tone-matched to the picture), the picture colours everything it can see")
        if skin_px:
            self.notes.append(f"mesh-lock: {skin_px:,} picture pixels the mesh did not cover were added as thin relief patches"
                              + (f" ({uplift_px:,} of them towers the mesh left too short)" if uplift_px else ""))
        if not meshio.have_quadric():
            self.notes.append("decimation used vertex clustering (pip install fast-simplification for cleaner LODs)")
        if "box projection" in how_uv:
            self.notes.append("UVs came from the box-projection fallback (pip install xatlas for proper UV islands)")

    def set_back(self, back_img: Optional[np.ndarray], style: str):
        """Called by the pipeline once the optional back-view picture is registered (mirrored to the front framing)."""
        p = self.charts[0].params
        p["back_img"] = back_img
        p["style"] = style

    def source_alpha(self):
        return self.s.alpha

    def layers(self, lod: int = 0, max_faces: int = 0) -> List[Layer]:
        v, f, cuv, fc = self._lods[min(int(lod), len(self._lods) - 1)]
        return [Layer("mesh", v.astype(np.float64), f.astype(np.int64), cuv.astype(np.float32), fc.astype(np.int32),
                      role="body")]

    def collision_groups(self, layers):
        from .collision import kmeans_hulls
        pts = np.concatenate([L.verts for L in layers])
        return kmeans_hulls(pts, int(self.opts.get("collision_hulls", 16 if self.s.kind in ("arena", "island") else 8)))


from pathlib import Path  # noqa: E402  (used in the note above)
