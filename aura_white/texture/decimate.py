"""Mesh decimation for the texturing stage (target ~50-150k faces before UV unwrapping).

Order: fast_simplification (quadric edge collapse, compiled; run in a child process because a crashing
native module must not kill the pipeline) -> built-in quadric *vertex clustering* (NumPy only, always works)."""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def have_fast_simplification() -> bool:
    try:
        import fast_simplification  # noqa: F401

        return True
    except Exception:
        return False


def clean(verts: np.ndarray, faces: np.ndarray, weld: float = 1e-7):
    """Weld duplicate vertices, drop degenerate faces and unreferenced vertices."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if weld > 0:
        span = max(float(np.ptp(verts, axis=0).max()), 1e-12)
        q = np.round(verts / (span * weld)).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        verts, faces = verts[first], inv.reshape(-1)[faces]
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[ok]
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used].astype(np.float32), remap[faces]


def fast_simplification_worker(verts, faces, target):
    import fast_simplification as fs

    v, f = fs.simplify(np.ascontiguousarray(verts, dtype=np.float32), np.ascontiguousarray(faces, dtype=np.int32),
                       target_count=int(target), agg=5)
    return {"verts": np.asarray(v, dtype=np.float32), "faces": np.asarray(f, dtype=np.int64)}


def decimate(verts: np.ndarray, faces: np.ndarray, target_faces: int,
             backend: str = "auto") -> Tuple[np.ndarray, np.ndarray, str]:
    """Returns (verts, faces, backend_used). No-op when the mesh is already small enough."""
    verts, faces = clean(verts, faces)
    if len(faces) <= target_faces:
        return verts, faces, "none"
    if backend in ("auto", "quadric") and have_fast_simplification():
        try:
            from .._isolate import call_isolated

            d = call_isolated("aura_white.texture.decimate:fast_simplification_worker",
                              dict(verts=verts, faces=faces, target=target_faces), timeout=300 + len(faces) / 2000)
            v, f = d["verts"], d["faces"]
            if len(f) > 0 and f.max() < len(v) and np.isfinite(v).all():
                v, f = clean(v, f, weld=0)
                if len(f) >= 0.3 * target_faces:
                    return v, f, "fast_simplification"
        except Exception as e:
            print(f"[aura-white] fast_simplification unusable ({e}); using built-in clustering decimation")
    v, f = cluster_decimate(verts, faces, target_faces)
    return v, f, "builtin"


def cluster_decimate(verts: np.ndarray, faces: np.ndarray, target_faces: int, iters: int = 14):
    """Quadric-weighted vertex clustering (Lindstrom 2000): pick the grid cell size by bisection so the result
    has about `target_faces` faces, place each cluster vertex at the quadric minimiser (regularised towards the
    cluster centroid). Pure NumPy; keeps thin structures (blades) far better than uniform clustering."""
    t0 = time.time()
    v = verts.astype(np.float64)
    tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    n = n / np.maximum(2 * area, 1e-20)[:, None]
    d = -(n * tri[:, 0]).sum(1)
    plane = np.concatenate([n, d[:, None]], 1)                       # (F,4)
    q_face = area[:, None, None] * (plane[:, :, None] * plane[:, None, :])   # (F,4,4)
    q_vert = np.zeros((len(v), 4, 4))
    for k in range(3):
        np.add.at(q_vert, faces[:, k], q_face)
    lo_pt, hi_pt = v.min(0), v.max(0)
    span = float((hi_pt - lo_pt).max())

    def run(cell):
        g = np.floor((v - lo_pt) / cell).astype(np.int64)
        dims = g.max(0) + 1
        key = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
        _, cid = np.unique(key, return_inverse=True)
        cid = cid.reshape(-1)
        f = cid[faces]
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        return cid, f[keep], keep

    lo, hi = span / 4096.0, span / 2.0
    best = None                                   # (distance to target, cid, faces)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        cid, f, _keep = run(mid)
        nf = len(np.unique(np.sort(f, axis=1), axis=0)) if len(f) < 4_000_000 else len(f)
        if best is None or abs(nf - target_faces) < best[0]:
            best = (abs(nf - target_faces), cid, f, nf)
        if nf > target_faces:
            lo = mid
        else:
            hi = mid
    _, cid, f, _ = best
    nc = int(cid.max()) + 1
    q_cell = np.zeros((nc, 4, 4))
    np.add.at(q_cell, cid, q_vert)
    cnt = np.bincount(cid, minlength=nc).astype(np.float64)
    centroid = np.zeros((nc, 3))
    np.add.at(centroid, cid, v)
    centroid /= cnt[:, None]
    A = q_cell[:, :3, :3]
    b = -q_cell[:, :3, 3]
    lam = 1e-3 * np.maximum(np.trace(A, axis1=1, axis2=2), 1e-12)          # pull to the centroid when ill-posed
    A = A + lam[:, None, None] * np.eye(3)
    b = b + lam[:, None] * centroid
    x = np.linalg.solve(A, b[..., None])[..., 0]
    bad = ~np.isfinite(x).all(1) | (np.linalg.norm(x - centroid, axis=1) > 2.5 * (span / 64))
    x[bad] = centroid[bad]
    f = np.unique(np.sort(f, axis=1), axis=0) if len(f) < 4_000_000 else f
    # restore consistent winding (sorting lost it): orient every face like the original geometry
    orig_n = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    ref = np.zeros((nc, 3))
    vn = np.zeros((len(v), 3))
    for k in range(3):
        np.add.at(vn, faces[:, k], n * area[:, None])
    np.add.at(ref, cid, vn)
    flip = (orig_n * ref[f].sum(1)).sum(1) < 0
    f[flip] = f[flip][:, ::-1]
    used = np.unique(f)
    remap = np.full(nc, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    print(f"[aura-white] cluster decimation: {len(faces)} -> {len(f)} faces in {time.time() - t0:.1f}s")
    return x[used].astype(np.float32), remap[f]
