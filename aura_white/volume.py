"""Density-volume construction for meshing, with a coarse-to-fine mode.

Dense evaluation (what TripoSR does) sends resolution^3 points - 16.7 M at 256 - through the
MLP. The triplane field is smooth at ~4-voxel scale, so a coarse pass finds where the surface
can be and only that shell is evaluated at full resolution (typically 5-10x fewer queries).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _interp_matrix(n_fine: int, n_coarse: int) -> np.ndarray:
    """(n_fine, n_coarse) linear-interpolation matrix between two lattices spanning [0, 1]."""
    u = np.linspace(0.0, 1.0, n_fine) * (n_coarse - 1)
    i0 = np.clip(np.floor(u).astype(np.int64), 0, n_coarse - 2)
    w = (u - i0).astype(np.float32)
    m = np.zeros((n_fine, n_coarse), np.float32)
    m[np.arange(n_fine), i0] = 1.0 - w
    m[np.arange(n_fine), i0 + 1] += w
    return m


def upsample_trilinear(vol: np.ndarray, n: int) -> np.ndarray:
    out = vol.astype(np.float32)
    for axis in range(3):
        m = _interp_matrix(n, vol.shape[axis])
        out = np.moveaxis(np.tensordot(m, out, axes=([1], [axis])), 0, axis)
    return np.ascontiguousarray(out, dtype=np.float32)


def _dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        p = np.pad(out, 1)
        acc = np.zeros_like(out)
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                for dz in (0, 1, 2):
                    acc |= p[dx : dx + out.shape[0], dy : dy + out.shape[1], dz : dz + out.shape[2]]
        out = acc
    return out


def _lattice(idx: np.ndarray, n: int, radius: float) -> np.ndarray:
    """Integer lattice coordinates (M,3) -> world coordinates in [-radius, radius]."""
    return (idx.astype(np.float32) / (n - 1) * 2.0 - 1.0) * radius


def build_density_volume(
    density_fn: Callable[[np.ndarray], np.ndarray],
    resolution: int,
    radius: float,
    threshold: float,
    adaptive: bool = True,
    chunk: int = 262144,
    progress: Optional[Callable[[str, float], None]] = None,
):
    """density_fn: (N,3) float32 world points -> (N,) density (numpy in/out).
    Returns (volume float32 (R,R,R), stats dict)."""
    R = int(resolution)
    stats = {"queries": 0, "dense_queries": R ** 3, "adaptive": False}

    def run(points: np.ndarray) -> np.ndarray:
        stats["queries"] += points.shape[0]
        return density_fn(points).astype(np.float32)

    def dense_slabs(n: int) -> np.ndarray:
        vol = np.empty((n, n, n), np.float32)
        ax = np.arange(n)
        gy, gz = np.meshgrid(ax, ax, indexing="ij")
        yz = np.stack([gy.ravel(), gz.ravel()], axis=1)
        slab = max(1, chunk // (n * n))
        for x0 in range(0, n, slab):
            xs = np.arange(x0, min(n, x0 + slab))
            idx = np.concatenate(
                [np.repeat(xs, yz.shape[0])[:, None], np.tile(yz, (len(xs), 1))], axis=1
            )
            vol[xs] = run(_lattice(idx, n, radius)).reshape(len(xs), n, n)
            if progress:
                progress("evaluating density", (x0 + len(xs)) / n)
        return vol

    if not adaptive or R < 96:
        return dense_slabs(R), stats

    Rc = (R - 1) // 4 + 1  # coarse lattice ~ the triplane's own feature resolution
    coarse = dense_slabs(Rc)
    stats["adaptive"] = True

    # a coarse cell can hold the surface unless all 8 corners are clearly outside or clearly inside
    corners = [
        coarse[dx : Rc - 1 + dx, dy : Rc - 1 + dy, dz : Rc - 1 + dz]
        for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)
    ]
    cmax = np.maximum.reduce(corners)
    cmin = np.minimum.reduce(corners)
    lo, hi = threshold / 6.0, threshold * 6.0
    cells = _dilate((cmax > lo) & (cmin < hi), 1)

    cidx = np.minimum((np.arange(R) / (R - 1) * (Rc - 1)).astype(np.int64), Rc - 2)
    active = cells[np.ix_(cidx, cidx, cidx)]

    vol = upsample_trilinear(coarse, R)
    flat = np.flatnonzero(active.ravel())
    stats["active_fraction"] = flat.size / float(R ** 3)
    for s in range(0, flat.size, chunk):
        part = flat[s : s + chunk]
        idx = np.stack(np.unravel_index(part, (R, R, R)), axis=1)
        vol.ravel()[part] = run(_lattice(idx, R, radius))
        if progress:
            progress("refining surface shell", min(1.0, (s + chunk) / max(flat.size, 1)))
    return vol, stats


def grid_to_world(v: np.ndarray, n: int, radius: float, pad: int = 0) -> np.ndarray:
    """Vertex positions in (possibly padded) grid-index units -> world units in [-radius, radius]."""
    return (((np.asarray(v, dtype=np.float32) - pad) / (n - 1)) * 2.0 - 1.0) * radius


def build_volume(density_fn, radius: float, resolution: int, threshold: float, adaptive: bool = True,
                 chunk: int = 262144, progress=None):
    """Same as build_density_volume; returns (volume, info) with info['mode'] in {'dense','adaptive'}
    and info['fraction'] = share of the full grid that was actually evaluated."""
    vol, st = build_density_volume(density_fn, resolution, radius, threshold, adaptive=adaptive, chunk=chunk,
                                   progress=progress)
    info = dict(st, mode="adaptive" if st["adaptive"] else "dense", fraction=st["queries"] / float(resolution ** 3))
    return vol, info
