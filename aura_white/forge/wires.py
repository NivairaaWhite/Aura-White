"""Hairline structures: find them in an image, turn them into real 3-D geometry.

Neural image-to-3D models lose chains, cords, whiskers and antennae because a 2-px wire never reaches the
iso-surface threshold of a voxel grid and its interlocking topology cannot be inferred from one view.  Here the wire
is detected in the artwork (thin, elongated mask component -> skeleton -> polyline) and built directly:

  * `chain`  interlocking, hollow stadium links (alternating orientation, real holes, links pass through each other)
  * `tube`   a round cord / vine / wire with a circular cross-section

NumPy + SciPy only.  Lengths inside the *_mesh functions are in whatever unit the polyline uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import morph


@dataclass
class Wire:
    pts: np.ndarray                 # (n,2) px (x, y), smoothed, ordered from the anchored end to the free end
    width: float                    # coverage width in px (what the eye sees, halo excluded)
    kind: str                       # "chain" | "tube"
    color: np.ndarray               # (3,) float 0..1 core colour
    anchored: bool = False          # start touches the bulk of the object
    length: float = 0.0
    mask: Optional[np.ndarray] = None   # full-frame bool mask of the component this wire was found in
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- skeleton graph
_N8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _crossing_number(sk: np.ndarray) -> np.ndarray:
    """Number of 0->1 transitions around each pixel (1 = end point, 2 = ordinary, >=3 = junction)."""
    p = np.pad(sk.astype(np.int8), 1)
    h, w = sk.shape
    nb = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _N8]
    cn = np.zeros(sk.shape, np.int8)
    for i in range(8):
        cn += ((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int8)
    return np.where(sk, cn, 0)


def trace_skeleton(sk: np.ndarray, min_len: int = 3) -> List[np.ndarray]:
    """Split a 1-px skeleton at junctions and end points into ordered pixel paths [(n,2) (x, y)]."""
    if not sk.any():
        return []
    cn = _crossing_number(sk)
    ys, xs = np.nonzero(sk)
    pix = set(zip(ys.tolist(), xs.tolist()))
    node = {(y, x) for y, x in pix if cn[y, x] != 2}
    if not node:                                     # closed loop: cut anywhere
        node = {(int(ys[0]), int(xs[0]))}
    visited = set()
    paths: List[List[Tuple[int, int]]] = []

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in _N8 if (y + dy, x + dx) in pix]

    for start in sorted(node):
        for first in nbrs(start):
            if first in visited and first not in node:
                continue
            path = [start, first]
            prev, cur = start, first
            visited.add(first)
            while cur not in node:
                cand = [q for q in nbrs(cur) if q != prev and q not in visited]
                if not cand:
                    break
                # prefer 4-adjacent continuation so that staircase corners do not fork
                cand.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
                nxt = cand[0]
                visited.add(nxt)
                path.append(nxt)
                prev, cur = cur, nxt
            if len(path) >= min_len:
                paths.append(path)
    out = []
    for pth in paths:
        a = np.asarray(pth, np.float64)
        out.append(a[:, ::-1].copy())                # (x, y)
    return out


def _smooth_resample(pts: np.ndarray, sigma: float, step: float) -> np.ndarray:
    from scipy import ndimage as ndi

    if len(pts) < 3:
        return pts
    q = np.stack([ndi.gaussian_filter1d(pts[:, k], sigma, mode="nearest") for k in range(2)], 1)
    q[0], q[-1] = pts[0], pts[-1]
    seg = np.hypot(*np.diff(q, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-6:
        return q[:1]
    n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)


# ----------------------------------------------------------------------------------------------- detection


def _fill_small_holes(comp: np.ndarray, max_area: int) -> np.ndarray:
    from scipy import ndimage as ndi

    holes = ndi.binary_fill_holes(comp) & ~comp
    if not holes.any():
        return comp
    lab, n = ndi.label(holes)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    small = sizes <= max_area
    small[0] = False
    return comp | small[lab]


def prune_spurs(sk: np.ndarray, min_len: int, rounds: int = 2) -> np.ndarray:
    """Delete short dead-end branches (medial-axis whiskers at link corners, ragged edges)."""
    sk = sk.copy()
    for _ in range(rounds):
        cn = _crossing_number(sk)
        changed = False
        for path in trace_skeleton(sk, min_len=2):
            xs, ys = path[:, 0].astype(int), path[:, 1].astype(int)
            c0, c1 = cn[ys[0], xs[0]], cn[ys[-1], xs[-1]]
            if len(path) < min_len and ((c0 == 1 and c1 >= 3) or (c1 == 1 and c0 >= 3)):
                keep = 0 if c0 >= 3 else len(path) - 1                 # keep the junction pixel
                for k in range(len(path)):
                    if k != keep:
                        sk[ys[k], xs[k]] = False
                changed = True
        if not changed:
            break
    return sk


def find_wires(mask: np.ndarray, rgb: np.ndarray, bg: Optional[np.ndarray] = None, max_radius: Optional[float] = None,
               min_length: Optional[float] = None, max_radius_cv: float = 0.65,
               **_ignored) -> Tuple[List[Wire], np.ndarray]:
    """Detect free-hanging hairline structures (chains, cords, antennae) in `mask` (bool, image space).

    Method: skeleton of the whole mask; skeleton pixels whose inscribed radius is small (<= `max_radius`) form
    "thin" branches; a connected thin branch that is long enough and has a roughly constant radius is a wire.
    This does not depend on morphological opening, so a chain whose links are alternately 5 and 12 px wide is found
    as one piece.  Returns (wires, wire_pixels); `wire_pixels` is the part of the mask the wires account for.
    `rgb` (uint8) supplies colours, `bg` (3,) the flat background colour (default white)."""
    from scipy import ndimage as ndi

    mask = np.asarray(mask, bool)
    H, W = mask.shape
    scale = float(max(H, W))
    r_max = float(max_radius if max_radius is not None else max(3.0, 0.0056 * scale))
    min_len = float(min_length if min_length is not None else max(24.0, 0.026 * scale))
    filled = _fill_small_holes(mask, max_area=int(max(30, 3 * r_max * r_max * 4)))
    # a ragged matte edge makes a noisy skeleton (dozens of whiskers): smooth the silhouette first
    smooth = ndi.gaussian_filter(filled.astype(np.float32), max(1.2, 0.3 * r_max)) >= 0.5
    smooth = _fill_small_holes(smooth, max_area=int(max(30, 3 * r_max * r_max * 4)))
    dt = morph.edt(smooth)
    sk = morph.thin(smooth)
    sk = prune_spurs(sk, int(max(6, 1.8 * r_max)), rounds=3)
    thin_sk = sk & (dt <= r_max)
    thin_sk = ndi.binary_closing(thin_sk, morph.disk(2.0)) & ndi.binary_dilation(sk, iterations=1)
    lab, n = morph.label(thin_sk)
    if n == 0:
        return [], np.zeros_like(mask)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    slices = ndi.find_objects(lab)
    bgc = np.asarray(bg if bg is not None else (255.0, 255.0, 255.0), np.float32)
    rgbf = np.asarray(rgb[..., :3], np.float32)
    bulk_dist = morph.edt(~(filled & (dt > r_max * 1.05)))          # distance to anything that is not a wire
    wires: List[Wire] = []
    used = np.zeros_like(mask)
    # nearest thin-skeleton component for every mask pixel (to know which pixels a wire accounts for)
    keep_ids = [i for i in range(1, n + 1) if sizes[i] >= 0.8 * min_len]
    if not keep_ids:
        return [], used
    good = np.isin(lab, keep_ids)
    iy, ix = morph.nearest_index(good)
    owner = lab[iy, ix]
    rad_here = dt[iy, ix]
    reach = filled & (morph.edt(~good) <= rad_here * 1.45 + 1.0)
    for i in keep_ids:
        sl = slices[i - 1]
        pad = int(r_max) + 3
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        csk = lab[y0:y1, x0:x1] == i
        rr = dt[y0:y1, x0:x1][csk]
        L = float(csk.sum())
        if L < min_len:
            continue
        cv = float(rr.std() / max(rr.mean(), 1e-6))
        if cv > max_radius_cv:
            continue
        region = reach[y0:y1, x0:x1] & (owner[y0:y1, x0:x1] == i)
        c = rgbf[y0:y1, x0:x1]
        dbg = np.linalg.norm(c - bgc, axis=2)
        core_d = np.percentile(dbg[region], 75) if region.any() else 0.0
        core_sel = region & (dbg >= core_d)
        core = np.median(c[core_sel], axis=0) if core_sel.any() else c[region].mean(0)
        denom = max(float(np.linalg.norm(core - bgc)), 1.0)
        cov = np.clip(dbg / denom, 0.0, 1.0) * region
        width_cov = float(cov.sum() / max(L, 1.0))
        width_fat = float(region.sum() / max(L, 1.0))
        rgb01 = np.clip(core / 255.0, 0, 1)
        mx, mn = float(rgb01.max()), float(rgb01.min())
        sat = 0.0 if mx <= 1e-6 else (mx - mn) / mx
        kind = "chain" if (sat < 0.30 and 0.12 < mx < 0.9) else "tube"
        for path in trace_skeleton(csk, min_len=3):
            plen = float(np.hypot(*np.diff(path, axis=0).T).sum())
            if plen < min_len:
                continue
            path = path + np.array([x0, y0], np.float64)
            d0 = bulk_dist[int(round(path[0, 1])), int(round(path[0, 0]))]
            d1 = bulk_dist[int(round(path[-1, 1])), int(round(path[-1, 0]))]
            if d1 < d0:
                path = path[::-1]
                d0, d1 = d1, d0
            step = max(1.0, 0.5 * max(width_fat, 2.0))
            pts = _smooth_resample(path, sigma=max(1.5, 0.5 * width_fat), step=step)
            if len(pts) < 3:
                continue
            # tapered tips of ribbons / fins are not wires: they belong to a sheet (radius shrinks towards the end)
            rp = dt[np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1), np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)]
            k3 = max(1, len(rp) // 3)
            taper = float(np.median(rp[-k3:]) / max(np.median(rp[:k3]), 1e-6))
            if taper < 0.55 or taper > 1.8:
                continue
            wires.append(Wire(pts, max(width_cov, 1.2), kind, rgb01, anchored=bool(d0 <= 2.5 * width_fat + 3.0),
                              length=float(np.hypot(*np.diff(pts, axis=0).T).sum()),
                              info={"fat_width": width_fat, "skeleton_px": L, "radius_cv": cv, "taper": taper}))
        used[y0:y1, x0:x1] |= region
    return wires, used


# ----------------------------------------------------------------------------------------------- geometry
def _ring(n: int) -> Tuple[np.ndarray, np.ndarray]:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.cos(a), np.sin(a)


def _tube_faces(rows: int, n_ring: int, closed_rows: bool) -> np.ndarray:
    f = []
    m = rows if closed_rows else rows - 1
    for i in range(m):
        j = (i + 1) % rows
        for k in range(n_ring):
            k2 = (k + 1) % n_ring
            a, b, c, d = i * n_ring + k, i * n_ring + k2, j * n_ring + k2, j * n_ring + k
            f.append((a, b, c))
            f.append((a, c, d))
    return np.asarray(f, np.int64)


def link_mesh(centre: np.ndarray, e1: np.ndarray, e2: np.ndarray, half_len: float, end_radius: float,
              wire_radius: float, n_arc: int = 7, n_ring: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """One hollow stadium link.  The link lies in the plane (e1, e2) around `centre`; e1 is the chain direction.
    `half_len` = half the straight part, `end_radius` = centre-line radius of the two end arcs."""
    th_r = np.linspace(np.pi / 2, -np.pi / 2, n_arc)
    th_l = np.linspace(-np.pi / 2, -3 * np.pi / 2, n_arc)
    px = np.r_[half_len + end_radius * np.cos(th_r), -half_len + end_radius * np.cos(th_l)]
    py = np.r_[end_radius * np.sin(th_r), end_radius * np.sin(th_l)]
    nx = np.r_[np.cos(th_r), np.cos(th_l)]
    ny = np.r_[np.sin(th_r), np.sin(th_l)]
    n_pl = np.cross(e1, e2)
    n_pl = n_pl / max(np.linalg.norm(n_pl), 1e-12)
    ck, sk = _ring(n_ring)
    pos = centre[None, :] + px[:, None] * e1[None, :] + py[:, None] * e2[None, :]
    out = nx[:, None] * e1[None, :] + ny[:, None] * e2[None, :]
    verts = (pos[:, None, :] + wire_radius * (ck[None, :, None] * out[:, None, :] + sk[None, :, None] * n_pl[None, None, :]))
    verts = verts.reshape(-1, 3)
    faces = _tube_faces(len(px), n_ring, closed_rows=True)
    return verts, faces


def _polyline_frames(P: np.ndarray):
    """Arclength parameter and unit tangents of a 3-D polyline."""
    seg = np.diff(P, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    s = np.r_[0.0, np.cumsum(ln)]
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0], T[-1] = seg[0], seg[-1]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    return s, T


def chain_mesh(P: np.ndarray, width: float, view_dir: np.ndarray, pitch_ratio: float = 1.3,
               n_arc: int = 6, n_ring: int = 6) -> Tuple[np.ndarray, np.ndarray, int]:
    """Interlocking chain along the 3-D polyline P (N,3); `width` = outer width of a face-on link (same unit as P);
    `view_dir` = unit vector from the camera into the scene.  Returns (verts, faces, n_links)."""
    P = np.asarray(P, np.float64)
    s, T = _polyline_frames(P)
    total = float(s[-1])
    r_w = 0.15 * width
    r_c = 0.5 * width - r_w
    pitch = pitch_ratio * width
    half_len = max(0.5 * (pitch - (2 * r_c - 2 * r_w)), 0.05 * width)
    n_links = int(total / pitch)
    if n_links < 1:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), 0
    V = np.asarray(view_dir, np.float64)
    V = V / max(np.linalg.norm(V), 1e-12)
    vs, fs = [], []
    base = 0
    for i in range(n_links):
        si = (i + 0.5) * pitch + 0.5 * (total - n_links * pitch)
        c = np.array([np.interp(si, s, P[:, k]) for k in range(3)])
        t = np.array([np.interp(si, s, T[:, k]) for k in range(3)])
        t /= max(np.linalg.norm(t), 1e-12)
        b = np.cross(V, t)
        if np.linalg.norm(b) < 1e-6:
            b = np.cross(np.array([0.0, 1.0, 0.0]), t)
        b /= np.linalg.norm(b)
        e2 = b if i % 2 == 0 else np.cross(t, b)         # face-on, then edge-on
        v, f = link_mesh(c, t, e2, half_len, r_c, r_w, n_arc, n_ring)
        vs.append(v)
        fs.append(f + base)
        base += len(v)
    return np.concatenate(vs, 0), np.concatenate(fs, 0), n_links


def tube_mesh(P: np.ndarray, radius: float, n_ring: int = 6, taper: float = 0.0,
              cap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Round tube around a 3-D polyline with parallel-transport frames; `taper` (0..1) shrinks the free end."""
    P = np.asarray(P, np.float64)
    s, T = _polyline_frames(P)
    n = len(P)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(T[0], ref)
    u /= np.linalg.norm(u)
    us, vs_ = [], []
    for i in range(n):
        if i:
            u = u - np.dot(u, T[i]) * T[i]
            nu = np.linalg.norm(u)
            u = u / nu if nu > 1e-9 else np.cross(T[i], ref)
        us.append(u.copy())
        vs_.append(np.cross(T[i], u))
    us, vs_ = np.asarray(us), np.asarray(vs_)
    ck, sk = _ring(n_ring)
    rad = radius * (1.0 - taper * (s / max(s[-1], 1e-9)))
    verts = P[:, None, :] + rad[:, None, None] * (ck[None, :, None] * us[:, None, :] + sk[None, :, None] * vs_[:, None, :])
    verts = verts.reshape(-1, 3)
    faces = _tube_faces(n, n_ring, closed_rows=False)
    if cap:
        c0, c1 = len(verts), len(verts) + 1
        verts = np.concatenate([verts, P[:1], P[-1:]], 0)
        cap0 = [(c0, (k + 1) % n_ring, k) for k in range(n_ring)]
        last = (n - 1) * n_ring
        cap1 = [(c1, last + k, last + (k + 1) % n_ring) for k in range(n_ring)]
        faces = np.concatenate([faces, np.asarray(cap0 + cap1, np.int64)], 0)
    return verts, faces
