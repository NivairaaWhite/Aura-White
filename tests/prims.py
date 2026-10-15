"""Numpy-only mesh primitives for tests (no trimesh needed).  All meshes are closed, outward-facing, CCW."""
from __future__ import annotations

import numpy as np


def uv_sphere(n_lat: int = 24, n_lon: int = 48, radius=1.0):
    """Closed UV sphere; `radius` may be a scalar or a 3-vector (ellipsoid)."""
    r = np.broadcast_to(np.asarray(radius, np.float64), (3,))
    th = np.linspace(0, np.pi, n_lat + 1)[1:-1]
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    ring = np.stack([np.sin(T) * np.cos(P), np.cos(T), np.sin(T) * np.sin(P)], -1).reshape(-1, 3)
    v = np.vstack([[0, 1, 0], ring, [0, -1, 0]]) * r
    f = []
    L = n_lat - 1
    idx = lambda i, j: 1 + i * n_lon + (j % n_lon)
    for j in range(n_lon):
        f.append([0, idx(0, j + 1), idx(0, j)])
    for i in range(L - 1):
        for j in range(n_lon):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            f += [[a, b, c], [a, c, d]]
    bot = len(v) - 1
    for j in range(n_lon):
        f.append([bot, idx(L - 1, j), idx(L - 1, j + 1)])
    return v.astype(np.float64), np.asarray(f, np.int64)


def box(size=(1.0, 1.0, 1.0), subdiv: int = 1):
    """Closed box centred on the origin, each face split into subdiv x subdiv quads."""
    sx, sy, sz = (np.asarray(size, np.float64) / 2.0)
    verts, faces = [], []

    def face(origin, du, dv):
        base = len(verts)
        for i in range(subdiv + 1):
            for j in range(subdiv + 1):
                verts.append(origin + du * (i / subdiv) + dv * (j / subdiv))
        for i in range(subdiv):
            for j in range(subdiv):
                a = base + i * (subdiv + 1) + j
                b, c, d = a + 1, a + subdiv + 2, a + subdiv + 1
                faces.extend([[a, c, b], [a, d, c]])          # (du x dv) points outward

    o = np.array
    face(o([-sx, -sy, sz]), o([2 * sx, 0, 0]), o([0, 2 * sy, 0]))      # +z
    face(o([sx, -sy, -sz]), o([-2 * sx, 0, 0]), o([0, 2 * sy, 0]))     # -z
    face(o([sx, -sy, sz]), o([0, 0, -2 * sz]), o([0, 2 * sy, 0]))      # +x
    face(o([-sx, -sy, -sz]), o([0, 0, 2 * sz]), o([0, 2 * sy, 0]))     # -x
    face(o([-sx, sy, sz]), o([2 * sx, 0, 0]), o([0, 0, -2 * sz]))      # +y
    face(o([-sx, -sy, -sz]), o([2 * sx, 0, 0]), o([0, 0, 2 * sz]))     # -y
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    key = np.round(v / max(float(np.abs(v).max()), 1e-9) * 1e6).astype(np.int64)      # weld the duplicated edge vertices
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return v[first], np.asarray(inv).reshape(-1)[f]


def merge(*meshes):
    vs, fs, off = [], [], 0
    for v, f in meshes:
        vs.append(np.asarray(v, np.float64))
        fs.append(np.asarray(f, np.int64) + off)
        off += len(v)
    return np.vstack(vs), np.vstack(fs)


def translate(m, t):
    return m[0] + np.asarray(t, np.float64), m[1]


def scale(m, s):
    return m[0] * np.asarray(s, np.float64), m[1]


def flip_orientation(m):
    return m[0], m[1][:, ::-1].copy()


def volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def figure():
    """Asymmetric standing figure (y up, front +z): body ellipsoid, a box arm to +x, a head, a small nose toward +z."""
    body = uv_sphere(28, 56, (0.45, 0.8, 0.30))
    arm = translate(box((0.7, 0.12, 0.12), 6), (0.55, 0.25, 0.0))
    head = translate(uv_sphere(18, 36, 0.22), (0.0, 1.0, 0.0))
    nose = translate(uv_sphere(10, 20, 0.07), (0.0, 1.0, 0.24))
    tail = translate(uv_sphere(12, 24, (0.07, 0.07, 0.30)), (-0.1, -0.4, -0.45))     # sticks out BEHIND: invisible from the front
    return merge(body, arm, head, nose, tail)


def slab(width=1.2, height=1.6, thickness=0.06):
    """A flat plate - what the old relief / source-locked mode produces."""
    return box((width, height, thickness), 8)
