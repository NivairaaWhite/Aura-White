"""UV unwrapping.  xatlas (best quality, wheels exist for most Pythons) when available, otherwise a built-in
box/cluster atlas that needs only NumPy - so texturing keeps working on any interpreter.

Contract: unwrap() returns (verts, faces, uv, vmap) where vertices are duplicated along seams,
``vmap[i]`` is the source vertex of new vertex i, ``faces`` keeps the input face order and ``uv`` is in [0,1]
with v growing downwards (image rows / glTF convention)."""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np


_XATLAS_BROKEN = False


def have_xatlas() -> bool:
    try:
        import xatlas  # noqa: F401

        return True
    except Exception:
        return False


def unwrap(verts: np.ndarray, faces: np.ndarray, resolution: int = 2048, backend: str = "auto",
           padding: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Returns (verts, faces, uv, vmap, backend_used)."""
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    pad = padding or max(2, resolution // 512)
    global _XATLAS_BROKEN
    if backend in ("auto", "xatlas") and have_xatlas() and (backend == "xatlas" or not _XATLAS_BROKEN):
        try:
            v, f, uv, vm = _xatlas_isolated(verts, faces, resolution, pad)
            return v, f, uv, vm, "xatlas"
        except Exception as e:
            _XATLAS_BROKEN = True
            if backend == "xatlas":
                raise
            print(f"[aura-white] xatlas unusable here ({e}); using the built-in unwrapper")
    elif backend == "xatlas":
        raise RuntimeError("xatlas is not installed (pip install xatlas) - or use --unwrap builtin")
    v, f, uv, vm = _box_unwrap(verts, faces, resolution, pad)
    return v, f, uv, vm, "builtin"


def xatlas_worker(verts, faces, resolution, pad):
    """Runs inside the child process (see aura_white._isolate)."""
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, faces.astype(np.uint32))
    co = xatlas.ChartOptions()
    co.max_iterations = 2
    po = xatlas.PackOptions()
    po.resolution = int(resolution)
    po.padding = int(pad)
    po.bilinear = True
    po.blockAlign = False
    po.bruteForce = len(faces) < 60_000          # brute force packing is slow on big meshes
    atlas.generate(co, po, verbose=False)
    vmap, idx, uv = atlas[0]
    return {"vmap": np.asarray(vmap, dtype=np.int64), "faces": np.asarray(idx, dtype=np.int64).reshape(-1, 3),
            "uv": np.asarray(uv, dtype=np.float32)}


def _xatlas_isolated(verts, faces, resolution, pad):
    """xatlas in a child process; the result is validated before it is trusted."""
    from .._isolate import call_isolated

    d = call_isolated("aura_white.texture.unwrap:xatlas_worker",
                      dict(verts=verts, faces=faces, resolution=resolution, pad=pad),
                      timeout=120 + len(faces) / 400.0)
    vmap, nf, uv = d["vmap"], d["faces"], d["uv"]
    if nf.shape != faces.shape or not np.isfinite(uv).all() or vmap.min() < 0 or vmap.max() >= len(verts):
        raise RuntimeError("xatlas returned an invalid atlas")
    if nf.max() >= len(vmap) or not np.allclose(verts[vmap][nf], verts[faces], atol=1e-6):
        raise RuntimeError("xatlas output does not match the input mesh")
    return verts[vmap], nf, np.clip(uv, 0.0, 1.0), vmap


# --------------------------------------------------------------------------------------------------
# built-in fallback
# --------------------------------------------------------------------------------------------------
_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
                 + [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)


def _face_adjacency(faces: np.ndarray):
    f = len(faces)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    fid = np.tile(np.arange(f), 3)
    key = np.sort(e, axis=1)
    key = key[:, 0].astype(np.int64) * (int(faces.max()) + 1) + key[:, 1]
    order = np.argsort(key, kind="stable")
    ks, fs = key[order], fid[order]
    same = ks[1:] == ks[:-1]
    a, b = fs[:-1][same], fs[1:][same]
    keep = a != b
    return a[keep], b[keep]


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lab = np.arange(n)
    if len(a) == 0:
        return lab
    for _ in range(64):
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]                              # pointer jumping
        if np.array_equal(new, lab):
            break
        lab = new
    _, inv = np.unique(lab, return_inverse=True)
    return inv


def _pack_shelves(sizes: np.ndarray, scale: float, pad: float):
    """Shelf-pack rectangles (w,h) scaled by `scale`, each padded by `pad` on every side, into the unit square.
    Returns (offsets, ok)."""
    w = sizes[:, 0] * scale + 2 * pad
    h = sizes[:, 1] * scale + 2 * pad
    order = np.argsort(-h, kind="stable")
    off = np.zeros((len(sizes), 2))
    x = y = shelf_h = 0.0
    for i in order:
        if w[i] > 1.0 + 1e-9:
            return off, False
        if x + w[i] > 1.0 + 1e-9:
            y += shelf_h
            x, shelf_h = 0.0, 0.0
        if y + h[i] > 1.0 + 1e-9:
            return off, False
        off[i] = (x + pad, y + pad)
        x += w[i]
        shelf_h = max(shelf_h, h[i])
    return off, True


def _box_unwrap(verts, faces, resolution, pad_px):
    t0 = time.time()
    v64 = verts.astype(np.float64)
    tri = v64[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(2 * area, 1e-20)[:, None]
    label = np.argmax(fn @ _DIRS.T, axis=1)
    a, b = _face_adjacency(faces)
    for _ in range(3):                                   # majority smoothing: fewer, larger charts
        votes = np.zeros((len(faces), len(_DIRS)))
        np.add.at(votes, (a, label[b]), area[b])
        np.add.at(votes, (b, label[a]), area[a])
        votes[np.arange(len(faces)), label] += area * 1.5
        label = np.argmax(votes, axis=1)
    same = label[a] == label[b]
    chart = _components(len(faces), a[same], b[same])
    nc = int(chart.max()) + 1

    # per-chart planar basis from the area-weighted mean normal
    mean_n = np.zeros((nc, 3))
    np.add.at(mean_n, chart, fn * area[:, None])
    mean_n /= np.maximum(np.linalg.norm(mean_n, axis=1, keepdims=True), 1e-12)
    helper = np.where(np.abs(mean_n[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    ax_u = np.cross(helper, mean_n)
    ax_u /= np.maximum(np.linalg.norm(ax_u, axis=1, keepdims=True), 1e-12)
    ax_v = np.cross(mean_n, ax_u)

    # new vertices: one per (chart, source vertex)
    cv = chart[:, None].repeat(3, 1).reshape(-1)
    sv = faces.reshape(-1)
    key = cv.astype(np.int64) * len(verts) + sv
    uk, inv = np.unique(key, return_inverse=True)
    v_chart, v_src = uk // len(verts), uk % len(verts)
    p = v64[v_src]
    uv = np.stack([(p * ax_u[v_chart]).sum(1), (p * ax_v[v_chart]).sum(1)], 1)

    # rotate each chart to its principal axis (tighter rectangles), then normalise to a chart-local origin
    order = np.argsort(v_chart, kind="stable")
    bounds = np.searchsorted(v_chart[order], np.arange(nc + 1))
    sizes = np.zeros((nc, 2))
    for c in range(nc):
        idx = order[bounds[c]:bounds[c + 1]]
        q = uv[idx]
        if len(q) >= 3:
            qc = q - q.mean(0)
            w, vec = np.linalg.eigh(qc.T @ qc + 1e-18 * np.eye(2))
            q = qc @ vec[:, ::-1]
        uv[idx] = q - q.min(0)
        sizes[c] = uv[idx].max(0) + 1e-9
    # global texel density: bisection on the largest scale that still packs into the unit square
    pad = pad_px / resolution
    lo, hi, best = 0.0, 1.0 / max(sizes.max(), 1e-9), None
    for _ in range(30):
        mid = (lo + hi) / 2
        off, ok = _pack_shelves(sizes, mid, pad)
        if ok:
            lo, best = mid, off
        else:
            hi = mid
    if best is None:
        raise RuntimeError("could not pack UV charts")
    uv = uv * lo + best[v_chart]
    uv = np.clip(uv, 0.0, 1.0).astype(np.float32)
    new_faces = inv.reshape(-1, 3).astype(np.int64)
    print(f"[aura-white] built-in unwrap: {nc} charts in {time.time() - t0:.1f}s")
    return verts[v_src], new_faces, uv, v_src.astype(np.int64)


def probe_backend(resolution: int = 256) -> str:
    """Which unwrapper actually works on this machine? Uses a ~3000 face sphere - tiny meshes can pass while a
    broken native build corrupts memory on realistic ones."""
    global _XATLAS_BROKEN
    n_u, n_v = 48, 32
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0.05, np.pi - 0.05, n_v)
    P = np.stack([np.outer(np.sin(v), np.cos(u)), np.outer(np.sin(v), np.sin(u)),
                  np.outer(np.cos(v), np.ones(n_u))], -1).reshape(-1, 3).astype(np.float32)
    F = []
    for j in range(n_v - 1):
        for i in range(n_u):
            a, b = j * n_u + i, j * n_u + (i + 1) % n_u
            c, d = (j + 1) * n_u + i, (j + 1) * n_u + (i + 1) % n_u
            F += [[a, c, b], [b, c, d]]
    _XATLAS_BROKEN = False
    return unwrap(P, np.array(F), resolution, "auto")[4]
