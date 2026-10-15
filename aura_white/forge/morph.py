"""Small image-morphology toolbox for the forge (NumPy + SciPy only; no scikit-image, no OpenCV).

Everything here works on boolean masks / float fields in image space (row 0 = top)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _ndi():
    try:
        from scipy import ndimage
        return ndimage
    except Exception as e:                                        # pragma: no cover - message only
        raise RuntimeError("the forge needs SciPy (pip install scipy) - it is the only extra package it uses") from e


def disk(r: float) -> np.ndarray:
    n = int(np.ceil(r))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return (yy * yy + xx * xx) <= r * r + 1e-6


def opening(mask: np.ndarray, r: float) -> np.ndarray:
    ndi = _ndi()
    se = disk(r)
    return ndi.binary_dilation(ndi.binary_erosion(mask, se, border_value=0), se)


def edt(mask: np.ndarray) -> np.ndarray:
    return _ndi().distance_transform_edt(mask)


def nearest_index(source: np.ndarray):
    """For every pixel the (row, col) of the nearest True pixel of `source`."""
    idx = _ndi().distance_transform_edt(~source, return_distances=False, return_indices=True)
    return idx[0], idx[1]


def label(mask: np.ndarray, connectivity: int = 2) -> Tuple[np.ndarray, int]:
    st = np.ones((3, 3), bool) if connectivity == 2 else None
    return _ndi().label(mask, structure=st)


# ---------------------------------------------------------------------------------------------------------
# thinning
# ---------------------------------------------------------------------------------------------------------
def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen skeleton, vectorised over the *boundary* pixels only (fast on large masks)."""
    h, w = mask.shape
    W = w + 2
    img = np.zeros((h + 2) * W, np.uint8)
    pad = np.zeros((h + 2, W), np.uint8)
    pad[1:-1, 1:-1] = mask
    img[:] = pad.ravel()
    off = np.array([-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1])            # p2..p9 (clockwise from north)
    fg = np.flatnonzero(img)
    nb = img[fg[:, None] + off[None, :]]
    cand = fg[nb.min(1) == 0]                                                # foreground next to background
    while cand.size:
        removed_any = False
        for step in (0, 1):
            if cand.size == 0:
                break
            nb = img[cand[:, None] + off[None, :]].astype(np.int16)
            p2, p3, p4, p5, p6, p7, p8, p9 = (nb[:, i] for i in range(8))
            B = nb.sum(1)
            A = ((p2 == 0) & (p3 == 1)).astype(np.int16) + ((p3 == 0) & (p4 == 1)) + ((p4 == 0) & (p5 == 1)) \
                + ((p5 == 0) & (p6 == 1)) + ((p6 == 0) & (p7 == 1)) + ((p7 == 0) & (p8 == 1)) \
                + ((p8 == 0) & (p9 == 1)) + ((p9 == 0) & (p2 == 1))
            if step == 0:
                c = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                c = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            rm = (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                gone = cand[rm]
                img[gone] = 0
                removed_any = True
                nbrs = (gone[:, None] + off[None, :]).ravel()
                nbrs = nbrs[img[nbrs] == 1]
                cand = np.unique(np.concatenate([cand[~rm], nbrs]))
        if not removed_any:
            break
    return img.reshape(h + 2, W)[1:-1, 1:-1].astype(bool)


# ---------------------------------------------------------------------------------------------------------
# marching squares: ordered sub-pixel contours of a scalar field
# ---------------------------------------------------------------------------------------------------------
# For every cell (corners TL TR BR BL) the segments are directed so that the inside (field >= level) is on the
# visual RIGHT of the direction of travel (image coordinates: x right, y down).  Edge letters: T B L R.
_SEG_TABLE = {
    1: [("L", "B")], 2: [("B", "R")], 3: [("L", "R")], 4: [("R", "T")], 6: [("B", "T")], 7: [("L", "T")],
    8: [("T", "L")], 9: [("T", "B")], 11: [("T", "R")], 12: [("R", "L")], 13: [("R", "B")], 14: [("B", "L")],
}
_SADDLE = {  # case -> (pairs when the cell centre is OUTSIDE, pairs when it is INSIDE)
    5: ([("R", "T"), ("L", "B")], [("L", "T"), ("R", "B")]),
    10: ([("T", "L"), ("B", "R")], [("T", "R"), ("B", "L")]),
}


def _edge(letter: str, y: np.ndarray, x: np.ndarray, W: int) -> np.ndarray:
    if letter == "T":
        return (y * W + x) * 2
    if letter == "B":
        return ((y + 1) * W + x) * 2
    if letter == "L":
        return (y * W + x) * 2 + 1
    return (y * W + x + 1) * 2 + 1


def contours(field: np.ndarray, level: float = 0.5) -> List[np.ndarray]:
    """Closed iso-contours of `field` at `level` as (n, 2) arrays of (x, y) pixel-centre coordinates.
    The inside (field >= level) lies on the visual right of the walking direction (y down), so outer boundaries
    have NEGATIVE shoelace area and holes POSITIVE area (see `polygon_area`)."""
    f = np.pad(np.asarray(field, np.float32), 1, constant_values=float(level) - 1.0)
    H, W = f.shape
    inside = f >= level
    case = inside[:-1, :-1] * 8 + inside[:-1, 1:] * 4 + inside[1:, 1:] * 2 + inside[1:, :-1] * 1
    ys, xs = np.nonzero((case > 0) & (case < 15))
    if ys.size == 0:
        return []
    cs = case[ys, xs]
    ea, eb = [], []
    for c, lst in _SEG_TABLE.items():
        sel = np.flatnonzero(cs == c)
        for a, b in lst:
            if sel.size:
                ea.append(_edge(a, ys[sel], xs[sel], W))
                eb.append(_edge(b, ys[sel], xs[sel], W))
    for c, (apart, joined) in _SADDLE.items():
        sel = np.flatnonzero(cs == c)
        if not sel.size:
            continue
        centre = (f[ys[sel], xs[sel]] + f[ys[sel], xs[sel] + 1] + f[ys[sel] + 1, xs[sel] + 1]
                  + f[ys[sel] + 1, xs[sel]]) / 4.0 >= level
        for group, pairs in ((sel[~centre], apart), (sel[centre], joined)):
            for a, b in pairs:
                if group.size:
                    ea.append(_edge(a, ys[group], xs[group], W))
                    eb.append(_edge(b, ys[group], xs[group], W))
    ea, eb = np.concatenate(ea), np.concatenate(eb)

    used = np.unique(np.concatenate([ea, eb]))
    cell, vert = used // 2, (used % 2) == 1
    ey, ex = cell // W, cell % W
    px = np.empty(len(used), np.float64)
    py = np.empty(len(used), np.float64)
    hy, hx = ey[~vert], ex[~vert]
    a0, a1 = f[hy, hx].astype(np.float64), f[hy, hx + 1].astype(np.float64)
    px[~vert], py[~vert] = hx + np.clip((level - a0) / np.where(a1 == a0, 1.0, a1 - a0), 0, 1), hy
    vy, vx = ey[vert], ex[vert]
    a0, a1 = f[vy, vx].astype(np.float64), f[vy + 1, vx].astype(np.float64)
    px[vert], py[vert] = vx, vy + np.clip((level - a0) / np.where(a1 == a0, 1.0, a1 - a0), 0, 1)

    slot = {int(e): i for i, e in enumerate(used.tolist())}
    nxt = np.full(len(used), -1, np.int64)
    for a, b in zip(ea.tolist(), eb.tolist()):
        nxt[slot[a]] = slot[b]
    visited = np.zeros(len(used), bool)
    loops: List[np.ndarray] = []
    for s in range(len(used)):
        if visited[s]:
            continue
        path, cur = [], s
        while cur >= 0 and not visited[cur]:
            visited[cur] = True
            path.append(cur)
            cur = nxt[cur]
        if len(path) >= 3 and cur == s:
            idx = np.asarray(path)
            loops.append(np.stack([px[idx] - 1.0, py[idx] - 1.0], 1))
    return loops


def polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def simplify_closed(xy: np.ndarray, tol: float, max_len: float = 0.0) -> np.ndarray:
    """Douglas-Peucker on a closed loop (anchored at the two mutually farthest-ish points), then optionally break
    edges longer than `max_len`.  Returns a subset of the input points plus the inserted ones."""
    n = len(xy)
    if n <= 4:
        return xy
    a = 0
    b = int(np.argmax(np.hypot(xy[:, 0] - xy[a, 0], xy[:, 1] - xy[a, 1])))
    keep = np.zeros(n, bool)
    keep[a] = keep[b] = True
    P = np.concatenate([xy, xy], 0)
    stack = [(a, b), (b, a + n)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        seg = P[i + 1:j]
        p, q = P[i], P[j]
        d = q - p
        L = float(np.hypot(d[0], d[1]))
        if L < 1e-9:
            dist = np.hypot(seg[:, 0] - p[0], seg[:, 1] - p[1])
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m % n] = True
            stack.append((i, m))
            stack.append((m, j))
    out = xy[keep]
    if max_len > 0 and len(out) >= 3:
        nxt = np.roll(out, -1, axis=0)
        L = np.hypot(*(nxt - out).T)
        k = np.floor(L / max_len).astype(int)
        if k.any():
            res = []
            for i in range(len(out)):
                res.append(out[i])
                for s in range(1, k[i] + 1):
                    res.append(out[i] + (nxt[i] - out[i]) * (s / (k[i] + 1)))
            out = np.asarray(res)
    return out
