"""Analytic bodies: UV sphere / ellipsoid, body of revolution, planar discs.  All in world space (y up, px units)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .layers import Chart, Layer


def _orient_outward(verts: np.ndarray, faces: np.ndarray, closed: bool = True) -> np.ndarray:
    if not closed:
        return faces
    v = verts
    vol = float(np.sum(np.einsum("ij,ij->i", v[faces[:, 0]], np.cross(v[faces[:, 1]], v[faces[:, 2]]))) / 6.0)
    return faces[:, [0, 2, 1]] if vol < 0 else faces


def uv_sphere(center: np.ndarray, radii: Tuple[float, float, float], n_lon: int, n_lat: int, chart: Chart,
              name: str = "sphere") -> Layer:
    """Ellipsoid whose pole axis is +y, longitude 0 facing the reference camera (+z).  Chart u = longitude (the seam
    is on the far side), v = latitude from the north pole; the chart is an equirectangular map."""
    rx, ry, rz = radii
    lat = np.linspace(0.0, np.pi, n_lat + 1)                 # 0 = north pole
    lon = np.arange(n_lon) * (2 * np.pi / n_lon) - np.pi      # -pi .. pi (exclusive); 0 faces the camera
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    X = rx * np.sin(LA) * np.sin(LO)
    Y = ry * np.cos(LA)
    Z = rz * np.sin(LA) * np.cos(LO)
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3) + np.asarray(center, np.float64)
    idx = np.arange((n_lat + 1) * n_lon).reshape(n_lat + 1, n_lon)
    faces, cuv = [], []
    for i in range(n_lat):
        for_j = np.arange(n_lon)
        j1 = (for_j + 1) % n_lon
        a, b = idx[i, for_j], idx[i, j1]
        c, d = idx[i + 1, for_j], idx[i + 1, j1]
        u0 = for_j / n_lon * chart.width
        u1 = (for_j + 1) / n_lon * chart.width
        v0 = np.full(n_lon, i / n_lat * chart.height)
        v1 = np.full(n_lon, (i + 1) / n_lat * chart.height)
        if i > 0:                                            # skip degenerate triangles at the north pole
            faces.append(np.stack([a, c, b], 1))
            cuv.append(np.stack([np.stack([u0, v0], 1), np.stack([u0, v1], 1), np.stack([u1, v0], 1)], 1))
        if i < n_lat - 1:                                    # ... and at the south pole
            faces.append(np.stack([b, c, d], 1))
            cuv.append(np.stack([np.stack([u1, v0], 1), np.stack([u0, v1], 1), np.stack([u1, v1], 1)], 1))
    faces = np.concatenate(faces).astype(np.int64)
    cuv = np.concatenate(cuv).astype(np.float32)
    faces2 = _orient_outward(verts, faces)
    if faces2 is not faces and not np.array_equal(faces2, faces):
        cuv = cuv[:, [0, 2, 1]]
    return Layer(name, verts, faces2, cuv, np.full(len(faces2), chart.cid, np.int32), role="body")


def revolve(profile: np.ndarray, n_theta: int, center_xz: Tuple[float, float], chart: Chart,
            y_offset: float = 0.0, name: str = "revolve", closed: bool = False,
            v_per_unit: float = 1.0, fold: bool = False, v0: float = 0.0,
            rho: Optional[np.ndarray] = None) -> Layer:
    """Surface of revolution about the vertical axis through `center_xz`.  `profile` is (K,2) of (rho, Y) from
    the bottom to the top.  Chart u = angle (0 = the side facing the camera at its centre column ... the seam is on the
    far side), v = distance along the profile in px.
    `fold`: angle theta and theta + pi share the same chart texels (the chart is half a turn wide); the faces of the
    two halves are tagged in ``face_side`` so the baker can tell them apart.
    `rho` (K, n_theta) overrides the radius of every vertex (non-circular footprints, left/right asymmetry); the chart v
    still follows `profile`."""
    prof = np.asarray(profile, np.float64)
    K = len(prof)
    th = np.arange(n_theta) * (2 * np.pi / n_theta) - np.pi
    rr = prof[:, 0][:, None] * np.ones((1, n_theta)) if rho is None else np.asarray(rho, np.float64)
    X = rr * np.sin(th)[None, :] + center_xz[0]
    Z = rr * np.cos(th)[None, :] + center_xz[1]
    Y = np.repeat((prof[:, 1] + y_offset)[:, None], n_theta, 1)
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    seg = np.hypot(np.diff(prof[:, 0]), np.diff(prof[:, 1]))
    s = v0 + np.r_[0.0, np.cumsum(seg)] * v_per_unit
    idx = np.arange(K * n_theta).reshape(K, n_theta)
    faces, cuv = [], []
    j = np.arange(n_theta)
    j1 = (j + 1) % n_theta
    if fold:
        assert n_theta % 2 == 0
        hh = n_theta // 2
        u0 = (j % hh) / hh * chart.width
        u1 = u0 + chart.width / hh
        side_j = (j >= hh).astype(np.int8)
    else:
        u0 = j / n_theta * chart.width
        u1 = (j + 1) / n_theta * chart.width
        side_j = np.zeros(n_theta, np.int8)
    sides, seg_k = [], []
    for k in range(K - 1):
        a, b = idx[k, j], idx[k, j1]
        c, d = idx[k + 1, j], idx[k + 1, j1]
        if prof[k, 0] > 1e-9:
            faces.append(np.stack([a, b, c], 1))
            sides.append(side_j)
            seg_k.append(np.full(n_theta, k))
            cuv.append(np.stack([np.stack([u0, np.full(n_theta, s[k])], 1), np.stack([u1, np.full(n_theta, s[k])], 1),
                                 np.stack([u0, np.full(n_theta, s[k + 1])], 1)], 1))
        if prof[k + 1, 0] > 1e-9:
            faces.append(np.stack([b, d, c], 1))
            sides.append(side_j)
            seg_k.append(np.full(n_theta, k))
            cuv.append(np.stack([np.stack([u1, np.full(n_theta, s[k])], 1), np.stack([u1, np.full(n_theta, s[k + 1])], 1),
                                 np.stack([u0, np.full(n_theta, s[k + 1])], 1)], 1))
    faces = np.concatenate(faces).astype(np.int64)
    cuv = np.concatenate(cuv).astype(np.float32)
    # outward = the profile tangent turned clockwise (rho right, Y up): a wall going up faces out, a rim going inwards faces
    # up, a slope going down and inwards faces up and inwards ...
    seg_k = np.concatenate(seg_k)
    tang = np.diff(prof, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    n_rho, n_y = tang[seg_k, 1], -tang[seg_k, 0]
    nrm = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    mid = verts[faces].mean(1)
    rad = np.stack([mid[:, 0] - center_xz[0], np.zeros(len(mid)), mid[:, 2] - center_xz[1]], 1)
    rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
    expect = rad * n_rho[:, None] + np.array([0.0, 1.0, 0.0]) * n_y[:, None]
    flip = np.sum(np.sum(nrm * expect, 1)) < 0
    if flip:
        faces = faces[:, [0, 2, 1]]
        cuv = cuv[:, [0, 2, 1]]
    return Layer(name, verts, faces, cuv, np.full(len(faces), chart.cid, np.int32), role="body",
                 info=dict(profile_length=float(s[-1] - v0)), face_side=np.concatenate(sides).astype(np.int8))


def disc(center_xyz: Tuple[float, float, float], radius: float, n_theta: int, n_rings: int, chart: Chart,
         up: bool = True, name: str = "disc", radial: Optional[np.ndarray] = None) -> Layer:
    """Horizontal disc with planar UVs (chart-local px: (x - cx + r, z - cz + r)).  Normal +y when `up`."""
    cx, cy, cz = center_xyz
    th = np.arange(n_theta) * (2 * np.pi / n_theta) - np.pi          # same angle convention as revolve(): x = sin, z = cos
    rings = np.linspace(0.0, radius, n_rings + 1)
    g = np.ones(n_theta) if radial is None else np.asarray(radial, np.float64)    # per-angle multiplier of the radius
    verts = [np.array([[cx, cy, cz]])]
    for r in rings[1:]:
        verts.append(np.stack([cx + r * g * np.sin(th), np.full(n_theta, cy), cz + r * g * np.cos(th)], 1))
    verts = np.concatenate(verts)
    faces = []
    for j in range(n_theta):
        faces.append([0, 1 + j, 1 + (j + 1) % n_theta])
    for k in range(1, n_rings):
        a0 = 1 + (k - 1) * n_theta
        a1 = 1 + k * n_theta
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            faces.append([a0 + j, a1 + j, a1 + jn])
            faces.append([a0 + j, a1 + jn, a0 + jn])
    faces = np.asarray(faces, np.int64)
    n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    if (n[:, 1].sum() > 0) != up:
        faces = faces[:, [0, 2, 1]]
    uvp = np.stack([verts[:, 0] - cx + radius, verts[:, 2] - cz + radius], 1) * (chart.width / (2 * radius))
    return Layer(name, verts, faces, uvp[faces].astype(np.float32), np.full(len(faces), chart.cid, np.int32), role="floor")
