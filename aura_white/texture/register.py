"""Camera registration by silhouette matching.

The texture of a generated mesh is only as sharp as the alignment between the mesh and the picture that is
projected onto it.  These routines recover the camera of the *original artwork* (and refine the cameras of
generated side views) by maximising the overlap (IoU) between the rendered mesh silhouette and the picture's
foreground mask - a derivative-free search that needs nothing but the rasteriser.

Parametrisation (mesh normalised to radius 1 at the origin):
    az, el   camera direction in degrees (az 0 = +x, the photo direction)
    logd     log camera distance;  logf log focal length (NDC);  cx, cy principal point offset (NDC)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..raster import Camera, Raster, orbit_camera

PARAMS = ("az", "el", "logd", "logf", "cx", "cy")
_MIN_DIST = 1.6


def make_camera(t: Dict[str, float]) -> Camera:
    d = max(math.exp(t["logd"]), _MIN_DIST)
    cam = orbit_camera(t["az"], t["el"], d, 30.0)
    return Camera(cam.w2c, math.exp(t["logf"]), t["cx"], t["cy"])


def _resize_mask(mask: np.ndarray, size: int) -> torch.Tensor:
    m = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    return (F.interpolate(m, size=(size, size), mode="area")[0, 0] > 0.5)


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / max(union, 1)


def _bbox_ndc(mask: torch.Tensor):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(xs) == 0:
        return None
    s = mask.shape[0]
    x0, x1 = xs.min().item(), xs.max().item() + 1
    y0, y1 = ys.min().item(), ys.max().item() + 1
    cx, cy = ((x0 + x1) / s) - 1.0, 1.0 - ((y0 + y1) / s)
    return cx, cy, (x1 - x0) / s * 2.0, (y1 - y0) / s * 2.0


@dataclass
class Registration:
    camera: Camera
    iou: float
    params: Dict[str, float]
    flipped: bool = False
    notes: List[str] = field(default_factory=list)


class _Fitter:
    def __init__(self, raster: Raster, verts: np.ndarray, faces: np.ndarray, target: np.ndarray, size: int = 160):
        self.r, self.size = raster, size
        self.v = torch.as_tensor(verts, dtype=torch.float32, device=raster.device)
        self.f = torch.as_tensor(faces, dtype=torch.long, device=raster.device)
        self.target = _resize_mask(target, size).to(raster.device)
        self.evals = 0

    def render(self, cam: Camera) -> torch.Tensor:
        self.evals += 1
        return self.r.rasterize_camera(self.v, self.f, cam, self.size).mask

    def score(self, t: Dict[str, float]) -> float:
        return _iou(self.render(make_camera(t)), self.target)

    # MAP prior: image->3D generators assume a roughly frontal, weakly perspective camera, and silhouettes barely
    # constrain az / el / distance (they trade against each other and against focal length), so without a
    # prior the search drifts along that valley and interior texture lands in the wrong place.
    prior: Dict[str, tuple] = {}
    prior_weight = 0.01             # IoU cost of a 1-sigma deviation

    def objective(self, t: Dict[str, float]) -> float:
        pen = sum((t[k] - mu) ** 2 / sg ** 2 for k, (mu, sg) in self.prior.items() if k in t)
        return self.score(t) - self.prior_weight * pen

    def fit_bbox(self, t: Dict[str, float]) -> Dict[str, float]:
        """Closed-form focal length / principal point so the silhouette bounding boxes coincide."""
        base = dict(t, logf=math.log(1.7), cx=0.0, cy=0.0)
        rb = _bbox_ndc(self.render(make_camera(base)))
        tb = _bbox_ndc(self.target)
        if rb is None or tb is None:
            return base
        s = math.sqrt(max(tb[2], 1e-3) / max(rb[2], 1e-3) * max(tb[3], 1e-3) / max(rb[3], 1e-3))
        out = dict(t)
        out["logf"] = base["logf"] + math.log(s)
        out["cx"], out["cy"] = tb[0] - s * rb[0], tb[1] - s * rb[1]
        return out

    def descend(self, t: Dict[str, float], free: Sequence[str], steps: Dict[str, float], levels: int = 5,
                sweeps: int = 3) -> Tuple[Dict[str, float], float]:
        t = dict(t)
        best = self.objective(t)
        steps = dict(steps)
        for _ in range(levels):
            for _ in range(sweeps):
                improved = False
                for name in free:
                    for sgn in (1.0, -1.0):
                        c = dict(t)
                        c[name] += sgn * steps[name]
                        if name == "logd":                 # dolly-zoom: keep the apparent size while the
                            c["logf"] += sgn * steps[name]  # perspective changes (the valley silhouettes leave open)
                        s = self.objective(c)
                        if s > best + 1e-4:
                            t, best, improved = c, s, True
                            break
                if not improved:
                    break
            steps = {k: v * 0.5 for k, v in steps.items()}
        return t, self.score(t)


_DEFAULT_STEPS = dict(az=6.0, el=6.0, logd=0.35, logf=0.08, cx=0.05, cy=0.05)


def register_reference(raster: Raster, verts: np.ndarray, faces: np.ndarray, mask: np.ndarray,
                       prior_az: float = 0.0, prior_el: float = 10.0, size: int = 160,
                       az_range: float = 24.0, allow_flip: bool = True, proxy_faces: int = 12000) -> Registration:
    """Camera of the original artwork.  `mask`: (H,W) bool foreground of the picture; `verts` normalised."""
    if len(faces) > proxy_faces:
        from .decimate import decimate

        verts, faces, _ = decimate(verts, faces, proxy_faces)
    def priors(base: float):
        return {"az": (prior_az + base, 12.0), "el": (prior_el, 12.0), "logd": (math.log(10.0), 0.4)}

    fit = _Fitter(raster, verts, faces, mask, size)
    notes: List[str] = []
    cands = []
    az_offsets = np.arange(-az_range, az_range + 1e-6, az_range / 2.0)
    flip_offsets = (180.0,) if allow_flip else ()
    for base in (0.0,) + flip_offsets:
        fit.prior = priors(base)
        for da in (az_offsets if base == 0.0 else (-12.0, 0.0, 12.0)):
            for el in (prior_el - 10, prior_el, prior_el + 12, prior_el + 24):
                for d in (5.0, 12.0):
                    t = dict(az=prior_az + base + da, el=el, logd=math.log(d), logf=0.5, cx=0.0, cy=0.0)
                    t = fit.fit_bbox(t)
                    cands.append((fit.objective(t) - (0.05 if base else 0.0), t, base))
    cands.sort(key=lambda c: -c[0])
    results = []
    for _, t, base in cands[:3]:
        fit.prior = priors(base)
        t2, s2 = fit.descend(t, PARAMS, _DEFAULT_STEPS)
        results.append((fit.objective(t2) - (0.03 if base else 0.0), s2, t2, base))
    results.sort(key=lambda r: -r[0])
    _, s, t, base = results[0]
    flipped = base != 0.0
    if size < 256:                                   # final polish at higher resolution, smaller steps
        fine = _Fitter(raster, verts, faces, mask, 256)
        fine.prior = priors(base)
        t, s = fine.descend(t, PARAMS, {k: v / 4 for k, v in _DEFAULT_STEPS.items()}, levels=3, sweeps=2)
    if flipped:
        notes.append("the reference seems to look at the BACK of the generated mesh (azimuth flipped by ~180 deg)")
    return Registration(make_camera(t), s, t, flipped, notes)


def align_view(raster: Raster, verts: np.ndarray, faces: np.ndarray, mask: np.ndarray, camera: Camera,
               size: int = 160, proxy_faces: int = 12000, refine_angles: bool = True) -> Registration:
    """Refine a nominal side view: focal length / principal point and (a few degrees of) azimuth / elevation."""
    if len(faces) > proxy_faces:
        from .decimate import decimate

        verts, faces, _ = decimate(verts, faces, proxy_faces)
    fit = _Fitter(raster, verts, faces, mask, size)
    pos = camera.position
    dist = float(np.linalg.norm(pos))
    az = math.degrees(math.atan2(pos[1], pos[0]))
    el = math.degrees(math.asin(np.clip(pos[2] / dist, -1, 1)))
    t = dict(az=az, el=el, logd=math.log(dist), logf=math.log(camera.f), cx=camera.cx, cy=camera.cy)
    t = fit.fit_bbox(t)
    steps = dict(_DEFAULT_STEPS, az=3.0, el=3.0)
    free = ("logf", "cx", "cy") + (("az", "el") if refine_angles else ())
    t, s = fit.descend(t, free, steps, levels=5)
    notes = []
    drift = abs(t["logf"] - math.log(camera.f))
    if drift > math.log(1.7) or abs(t["cx"]) > 0.4 or abs(t["cy"]) > 0.4:
        # a free zoom can make *any* picture "match" (blow the mesh up until it fills the frame): reject
        notes.append("alignment needed an implausible zoom/shift")
        s = 0.0
    return Registration(make_camera(t), s, t, False, notes)


# ---- 24 proper rotations (+ 24 mirrored) of the cube: recovery when a back-end delivers an unexpected frame ----
def cube_rotations(include_mirrors: bool = False) -> List[np.ndarray]:
    import itertools

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = s
            if np.linalg.det(m) > 0 or include_mirrors:
                out.append(m)
    return out


def find_orientation(raster: Raster, verts: np.ndarray, faces: np.ndarray, mask: np.ndarray, size: int = 128,
                     proxy_faces: int = 6000, allow_mirror: bool = False, confirm: int = 3) -> Tuple[np.ndarray, float]:
    """Best axis-aligned orientation by silhouette IoU: returns (M, iou), apply as verts @ M.T.
    All orientations are scored coarsely, the best few are then confirmed with the full registration search
    (a coarse score alone cannot separate near-identical silhouettes).  Mirrored candidates (det -1, which also need
    the face winding flipped) are only preferred when they beat the best proper rotation by a clear margin -
    a single silhouette cannot tell them apart otherwise."""
    if len(faces) > proxy_faces:
        from .decimate import decimate

        verts, faces, _ = decimate(verts, faces, proxy_faces)
    scored = []
    for R in cube_rotations(allow_mirror):
        proper = np.linalg.det(R) > 0
        f = faces if proper else faces[:, ::-1]
        fit = _Fitter(raster, verts @ R.T, f, mask, size)
        sc = max(fit.score(fit.fit_bbox(dict(az=0.0, el=el, logd=math.log(8.0), logf=0.5, cx=0.0, cy=0.0)))
                 for el in (0.0, 15.0))
        scored.append((sc, proper, R))
    best = {True: (None, -1.0), False: (None, -1.0)}
    for proper in (True, False):
        cands = sorted([c for c in scored if c[1] == proper], key=lambda c: -c[0])[:confirm]
        for _, _, R in cands:
            f = faces if proper else faces[:, ::-1]
            reg = register_reference(raster, verts @ R.T, f, mask, size=size, allow_flip=False, proxy_faces=proxy_faces)
            if reg.iou > best[proper][1]:
                best[proper] = (R, reg.iou)
    if best[False][0] is not None and best[False][1] > best[True][1] + 0.04:
        return best[False]
    return best[True]


# ---- foreground estimation for generated views (flat background) ----
def estimate_foreground(img: np.ndarray, tol: float = 0.07) -> np.ndarray:
    """(H,W,3) float [0,1] picture on a flat background -> bool foreground. Background = pixels close to the
    border colour that are connected to the border."""
    x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1)[None]
    border = torch.cat([x[0, :, 0], x[0, :, -1], x[0, :, :, 0], x[0, :, :, -1]], 1)
    bg = border.median(dim=1).values.view(1, 3, 1, 1)
    near = ((x - bg).abs().amax(1, keepdim=True) < tol).float()
    seed = torch.zeros_like(near)
    seed[..., 0, :] = seed[..., -1, :] = 1
    seed[..., :, 0] = seed[..., :, -1] = 1
    reach = seed * near
    for _ in range(max(x.shape[-2:])):
        nxt = F.max_pool2d(reach, 3, 1, 1) * near
        if torch.equal(nxt, reach):
            break
        reach = nxt
    return (reach[0, 0] < 0.5).numpy()


# ---------------------------------------------------------------------------------------------------------
# photometric refinement: align the artwork to the generated side views THROUGH the mesh
# ---------------------------------------------------------------------------------------------------------
class _PhotoFitter:
    """Silhouettes leave the camera's azimuth / elevation / distance under-determined, and every error there shifts
    interior texture (a 5 degree elevation error moves features by ~10% of the object).  The generated side views
    see the same surface from other directions, so the right artwork camera is the one for which artwork and side
    views agree about the colour of every surface point they both see (stereo consistency)."""

    def __init__(self, raster: Raster, verts, faces, normals, ref_rgb, ref_mask, gens, size: int = 176,
                 depth_res: int = 256):
        dev = raster.device
        self.r, self.S, self.dr = raster, size, depth_res
        self.V = torch.as_tensor(verts, dtype=torch.float32, device=dev)
        self.F = torch.as_tensor(faces, dtype=torch.long, device=dev)
        self.N = torch.as_tensor(normals, dtype=torch.float32, device=dev)
        ref = torch.as_tensor(np.ascontiguousarray(ref_rgb), dtype=torch.float32, device=dev).permute(2, 0, 1)[None]
        self.ref = F.avg_pool2d(F.interpolate(ref, size=(size, size), mode="area"), 3, 1, 1, count_include_pad=False)
        m = torch.as_tensor(ref_mask, dtype=torch.float32, device=dev)[None, None]
        m = F.interpolate(m, size=(size, size), mode="area")
        self.target = m[0, 0] > 0.5                       # for the silhouette term
        self.refmask = (-F.max_pool2d(-m, 5, 1, 2))[0, 0] > 0.5   # eroded: only trust pixels well inside
        self.gens = []
        for img, cam in gens:
            out = raster.rasterize_camera(self.V, self.F, cam, depth_res)
            depth = torch.where(torch.isfinite(out.depth), out.depth, torch.full_like(out.depth, 1e6))
            t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device=dev).permute(2, 0, 1)[None]
            t = F.avg_pool2d(t, 3, 1, 1, count_include_pad=False)
            self.gens.append(dict(img=t, dmin=-F.max_pool2d((-depth)[None, None], 3, 1, 1)[0, 0], cam=cam,
                                  m=torch.as_tensor(cam.w2c, dtype=torch.float32, device=dev),
                                  pos=torch.as_tensor(cam.position, dtype=torch.float32, device=dev)))
        self.evals = 0

    def evaluate(self, t: Dict[str, float]):
        """-> (photometric loss in [0,2], silhouette IoU). Lower loss is better."""
        from ..raster import interpolate

        self.evals += 1
        cam = make_camera(t)
        out = self.r.rasterize_camera(self.V, self.F, cam, self.S)
        iou = _iou(out.mask, self.target)
        hit = out.mask & self.refmask
        if int(hit.sum()) < 200:
            return 2.0, iou
        P = interpolate(self.V, self.F, out)[hit]
        Nn = F.normalize(interpolate(self.N, self.F, out)[hit], dim=-1)
        R = self.ref[0].permute(1, 2, 0)[hit]
        pos = torch.as_tensor(cam.position, dtype=torch.float32, device=P.device)
        to = pos - P
        cos_r = ((Nn * to).sum(-1) / to.norm(dim=-1).clamp(min=1e-9))
        tot, cnt = 0.0, 0
        for g in self.gens:
            pc = P @ g["m"][:3, :3].T + g["m"][:3, 3]
            w = -pc[:, 2]
            ws = w.clamp(min=1e-6)
            gc = g["cam"]
            gx = gc.f * pc[:, 0] / ws + gc.cx
            gy = -(gc.f * pc[:, 1] / ws + gc.cy)
            inb = (gx.abs() < 0.98) & (gy.abs() < 0.98) & (w > 1e-3)
            ix = ((gx * 0.5 + 0.5) * self.dr).long().clamp(0, self.dr - 1)
            iy = ((gy * 0.5 + 0.5) * self.dr).long().clamp(0, self.dr - 1)
            tg = g["pos"] - P
            cos_g = (Nn * tg).sum(-1) / tg.norm(dim=-1).clamp(min=1e-9)
            tan = (1 - cos_g ** 2).clamp(min=0).sqrt() / cos_g.clamp(min=1e-3)
            bias = 2.0 * w / (gc.f * self.dr) * (1.5 + 3.0 * tan.clamp(max=4.0))
            sel = inb & (w <= g["dmin"][iy, ix] + bias) & (cos_g > 0.4) & (cos_r > 0.4)
            n = int(sel.sum())
            if n < 300:
                continue
            grid = torch.stack([gx[sel], gy[sel]], -1).view(1, 1, -1, 2)
            G = F.grid_sample(g["img"], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
            X = R[sel]
            xm, gm = X - X.mean(0), G - G.mean(0)
            corr = (xm * gm).mean(0) / (xm.std(0) * gm.std(0) + 1e-4)
            tot += n * float(1.0 - corr.mean())
            cnt += n
        if cnt == 0:
            return 1.0, iou                      # nothing to compare against: neutral
        return tot / cnt, iou


def refine_reference_photometric(raster: Raster, verts: np.ndarray, faces: np.ndarray, normals: np.ndarray,
                                 ref_rgb: np.ndarray, ref_alpha: np.ndarray, gens, start: Registration,
                                 max_iou_loss: float = 0.03, size: int = 176) -> Registration:
    """Refine `start` (a silhouette registration) with the stereo-consistency objective.  Only accepted if it
    lowers the photometric loss clearly and the silhouette overlap does not degrade."""
    fit = _PhotoFitter(raster, verts, faces, normals, ref_rgb, ref_alpha > 0.5, gens, size)
    t = dict(start.params)
    lim = {"az": 10.0, "el": 10.0, "logd": 0.7, "logf": 0.4, "cx": 0.12, "cy": 0.12}
    t0 = dict(t)
    prior = {"az": (t0["az"], 12.0), "el": (t0["el"], 12.0), "logd": (math.log(10.0), 0.6)}

    def total(p):
        photo, iou = fit.evaluate(p)
        pen = sum((p[k] - mu) ** 2 / sg ** 2 for k, (mu, sg) in prior.items())
        return photo + 2.0 * (1.0 - iou) + 0.005 * pen, photo, iou

    best, photo0, iou0 = total(t)
    steps = dict(az=3.0, el=3.0, logd=0.25, logf=0.05, cx=0.03, cy=0.03)
    for _ in range(4):
        for _ in range(3):
            improved = False
            for name in PARAMS:
                for sgn in (1.0, -1.0):
                    c = dict(t)
                    c[name] += sgn * steps[name]
                    if name == "logd":
                        c["logf"] += sgn * steps[name]
                    if abs(c[name] - t0[name]) > lim[name]:
                        continue
                    s, _, _ = total(c)
                    if s < best - 1e-4:
                        t, best, improved = c, s, True
                        break
            if not improved:
                break
        steps = {k: v * 0.5 for k, v in steps.items()}
    _, photo1, iou1 = total(t)
    notes = list(start.notes)
    if photo1 < photo0 - 0.02 and iou1 > iou0 - max_iou_loss:
        notes.append(f"artwork camera refined by stereo consistency (photometric loss {photo0:.3f} -> {photo1:.3f})")
        return Registration(make_camera(t), iou1, t, start.flipped, notes)
    return start


def orientation_from_views(raster: Raster, verts: np.ndarray, faces: np.ndarray, masks: Sequence[np.ndarray],
                           cameras: Sequence[Camera], size: int = 128, proxy_faces: int = 8000,
                           improvement: float = 0.03) -> Tuple[np.ndarray, bool, float, float]:
    """Check (and if needed fix) the axis convention of a mesh that was reconstructed *from* known cameras.
    Tries all 48 signed axis permutations, mirrored ones included.  Returns (M, mirrored, best_iou, identity_iou)
    where the fixed mesh is  verts @ M.T  (reverse the face winding when `mirrored`)."""
    import itertools

    if len(faces) > proxy_faces:
        from .decimate import decimate

        verts, faces, _ = decimate(verts, faces, proxy_faces)
    targets = [_resize_mask(m, size).to(raster.device) for m in masks]
    f = torch.as_tensor(faces, dtype=torch.long, device=raster.device)

    def score(M, flip):
        v = torch.as_tensor(verts @ M.T, dtype=torch.float32, device=raster.device)
        ff = f.flip(1) if flip else f
        return float(np.mean([_iou(raster.rasterize_camera(v, ff, c, size).mask, t) for c, t in zip(cameras, targets)]))

    ident = score(np.eye(3), False)
    best = (np.eye(3), False, ident)
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (p_, sg) in enumerate(zip(perm, signs)):
                M[i, p_] = sg
            if np.allclose(M, np.eye(3)):
                continue
            mirrored = np.linalg.det(M) < 0
            sc = score(M, mirrored)
            if sc > best[2] + (improvement if np.allclose(best[0], np.eye(3)) else 0.0):
                best = (M, mirrored, sc)
    return best[0], best[1], best[2], ident
