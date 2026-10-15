"""Aura's own synthetic asset factory: analytic signed-distance assets with part colours and front/back differences.

Why this exists.  (1) It is the only training and test data that is 100 % ours - no licence questions - so every stage of
the native stack (VAE, flow model, verifier, self-improvement loop) can be exercised and *measured* anywhere, including
a laptop CPU.  (2) The ground truth is exact (analytic SDF), so "did the model get better?" has a true answer.
(3) It builds in the failure that started this project: the BACK of an asset is not a mirror of its FRONT (an emblem on
the front, stripes on the back, a cape or banner that exists only behind), so a model that mirrors, or regresses to the
mean, is caught by the numbers.

It is a curriculum and a test bench, not a substitute for real assets: real quality comes from real data (see NATIVE.md).

Frame (the package-wide convention): +y up, the front of the object faces +z, +x is the viewer's right at the front view.
Every asset is normalised to fit [-0.9, 0.9]^3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

FAMILIES = ("prop", "blade", "figure", "tower")


# ------------------------------------------------------------------------------------------------ primitives
def rot_euler(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    t = np.clip((wx * vx + wy * vy) / max(vx * vx + vy * vy, 1e-12), 0.0, 1.0)
    return np.hypot(wx - t * vx, wy - t * vy)


def prim_sdf(kind: str, p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Signed distance of primitive `kind` in its own frame (y is the long axis).  p: (N,3)."""
    if kind == "sphere":
        return np.linalg.norm(p, axis=1) - s[0]
    if kind in ("box", "rbox"):
        r = s[3] if kind == "rbox" else 0.0
        q = np.abs(p) - (s[:3] - r)
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0) - r
    if kind == "cylinder":
        d = np.stack([np.hypot(p[:, 0], p[:, 2]) - s[0], np.abs(p[:, 1]) - s[1]], 1)
        return np.minimum(d.max(1), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=1)
    if kind == "capsule":
        y = np.clip(p[:, 1], -s[1], s[1])
        return np.linalg.norm(np.stack([p[:, 0], p[:, 1] - y, p[:, 2]], 1), axis=1) - s[0]
    if kind == "torus":                                               # ring in the xz plane around the y axis
        q = np.stack([np.hypot(p[:, 0], p[:, 2]) - s[0], p[:, 1]], 1)
        return np.linalg.norm(q, axis=1) - s[1]
    if kind == "cone":                                                # base radius s[0] at y=-s[1], apex at y=+s[1]
        r, h = s[0], s[1]
        rho, y = np.hypot(p[:, 0], p[:, 2]), p[:, 1]
        d = np.minimum(_seg_dist(rho, y, 0.0, -h, r, -h), _seg_dist(rho, y, r, -h, 0.0, h))
        inside = (y > -h) & (y < h) & (rho < r * (h - y) / (2.0 * h))
        return np.where(inside, -d, d)
    raise ValueError(kind)


@dataclass
class Part:
    kind: str
    size: np.ndarray
    centre: np.ndarray
    R: np.ndarray
    colour: np.ndarray
    name: str = ""
    clip_n: Optional[np.ndarray] = None      # local half-space  dot(p, n) <= off  intersected with the primitive
    clip_off: float = 0.0
    only_back: bool = False                  # informational: the part lives behind the object

    def sdf(self, P: np.ndarray) -> np.ndarray:
        p = (P - self.centre) @ self.R
        d = prim_sdf(self.kind, p, self.size)
        if self.clip_n is not None:
            d = np.maximum(d, p @ self.clip_n - self.clip_off)
        return d

    def scaled(self, s: float, shift: np.ndarray) -> "Part":
        return Part(self.kind, self.size * s, self.centre * s + shift, self.R, self.colour, self.name, self.clip_n,
                    self.clip_off * s, self.only_back)


# ------------------------------------------------------------------------------------------------ assets
@dataclass
class ProcAsset:
    family: str
    seed: int
    parts: List[Part]
    accent: np.ndarray                      # front emblem colour
    stripe: np.ndarray                      # back stripe colour (a function of the accent - learnable, not a mirror)
    emblem_c: np.ndarray                    # (x, y) of the front emblem
    emblem_r: float
    stripe_freq: float
    stripe_phase: float
    meta: Dict = field(default_factory=dict)

    # -- geometry
    def sdf(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, np.float64).reshape(-1, 3)
        d = self.parts[0].sdf(P)
        for part in self.parts[1:]:
            d = np.minimum(d, part.sdf(P))
        return d

    def part_index(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, np.float64).reshape(-1, 3)
        ds = np.stack([part.sdf(P) for part in self.parts], 1)
        return ds.argmin(1)

    def colour(self, P: np.ndarray) -> np.ndarray:
        """Albedo at world points (N,3) in [0,1].  The front carries an emblem, the back stripes: the two sides differ."""
        P = np.asarray(P, np.float64).reshape(-1, 3)
        base = np.stack([p.colour for p in self.parts], 0)[self.part_index(P)]
        out = base.copy()
        front = P[:, 2] > 0.05
        back = P[:, 2] < -0.05
        emb = front & (np.hypot(P[:, 0] - self.emblem_c[0], P[:, 1] - self.emblem_c[1]) < self.emblem_r)
        out[emb] = self.accent
        stripes = back & (np.mod(P[:, 1] * self.stripe_freq + self.stripe_phase, 2.0) < 1.0)
        out[stripes] = 0.55 * out[stripes] + 0.45 * self.stripe
        return np.clip(out, 0.0, 1.0)

    def gradient(self, P: np.ndarray, eps: float = 2e-3) -> np.ndarray:
        P = np.asarray(P, np.float64).reshape(-1, 3)
        g = np.zeros_like(P)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            g[:, i] = self.sdf(P + e) - self.sdf(P - e)
        return g / (2 * eps)

    def normals(self, P: np.ndarray) -> np.ndarray:
        g = self.gradient(P)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)

    def _normalise(self) -> "ProcAsset":
        """Uniformly scale + centre the bounding box into [-0.9, 0.9]^3 (estimated from a coarse grid)."""
        n = 48
        c = (np.arange(n) + 0.5) / n * 2.4 - 1.2
        G = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
        inside = self.sdf(G) < 0.0
        if inside.sum() < 8:
            return self
        lo, hi = G[inside].min(0), G[inside].max(0)
        ext = float((hi - lo).max())
        s = 1.8 / max(ext, 1e-6)
        shift = -(lo + hi) / 2.0 * s
        self.parts = [p.scaled(s, shift) for p in self.parts]       # emblem / stripes are drawn in the final frame already
        return self

    # -- sampling
    def sample_surface(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """n surface points + normals (band rejection, then Newton projection onto the zero level set)."""
        got_p, got_n, have = [], [], 0
        for _ in range(8):
            m = max(n * 12, 4000)
            P = rng.uniform(-1.0, 1.0, size=(m, 3))
            d = self.sdf(P)
            keep = np.abs(d) < 0.09
            P = P[keep]
            for _it in range(4):
                if len(P) == 0:
                    break
                d = self.sdf(P)
                g = self.gradient(P)
                gn = np.maximum((g * g).sum(1, keepdims=True), 1e-9)
                P = P - d[:, None] * g / gn
            if len(P) == 0:
                continue
            ok = np.abs(self.sdf(P)) < 2.5e-3
            P = P[ok]
            got_p.append(P)
            have += len(P)
            if have >= n:
                break
        if have == 0:
            raise RuntimeError("asset has no surface")
        P = np.concatenate(got_p)
        idx = rng.choice(len(P), size=n, replace=len(P) < n)
        P = P[idx]
        return P.astype(np.float32), self.normals(P).astype(np.float32)

    def sample_queries(self, n: int, rng: np.random.Generator, surf: Optional[np.ndarray] = None):
        """Occupancy-style training queries: half uniform in the cube, half hugging the surface at three noise scales."""
        half = n // 2
        U = rng.uniform(-1.0, 1.0, size=(half, 3))
        if surf is None:
            surf, _ = self.sample_surface(max(half, 256), rng)
        base = surf[rng.integers(0, len(surf), size=n - half)].astype(np.float64)
        sig = rng.choice([0.006, 0.02, 0.06], size=(n - half, 1))
        Q = np.concatenate([U, base + rng.normal(size=base.shape) * sig])
        return Q.astype(np.float32), self.sdf(Q).astype(np.float32), self.colour(Q).astype(np.float32)

    def sdf_grid(self, res: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
        c = (np.arange(res) + 0.5) / res * (hi - lo) + lo
        G = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1).reshape(-1, 3)
        return self.sdf(G).reshape(res, res, res).astype(np.float32)

    def mesh(self, res: int = 64):
        """(verts, faces, vertex colours) of the zero level set (Aura's own surface-nets mesher)."""
        from .geometry import sdf_to_mesh

        v, f = sdf_to_mesh(self.sdf_grid(res))
        return v.astype(np.float64), f.astype(np.int64), self.colour(v).astype(np.float32)


# ------------------------------------------------------------------------------------------------ builders
_PALETTE = np.array([
    [0.78, 0.18, 0.16], [0.16, 0.34, 0.74], [0.18, 0.62, 0.30], [0.90, 0.72, 0.18], [0.55, 0.22, 0.66],
    [0.92, 0.50, 0.14], [0.14, 0.60, 0.66], [0.82, 0.82, 0.84], [0.30, 0.30, 0.34], [0.52, 0.34, 0.20],
    [0.88, 0.56, 0.62], [0.12, 0.14, 0.18],
])
_SKIN = np.array([[0.93, 0.76, 0.62], [0.80, 0.58, 0.42], [0.60, 0.40, 0.28], [0.96, 0.84, 0.74]])


def _col(rng, jitter: float = 0.05):
    c = _PALETTE[int(rng.integers(len(_PALETTE)))] + rng.normal(0, jitter, 3)
    return np.clip(c, 0.03, 0.97)


def _complement(c: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - c * 0.9 + 0.05, 0.0, 1.0)


def _mk(kind, size, centre, rot=(0, 0, 0), colour=None, name="", clip=None, only_back=False) -> Part:
    p = Part(kind, np.asarray(size, np.float64), np.asarray(centre, np.float64), rot_euler(*rot),
             np.asarray(colour if colour is not None else [0.6, 0.6, 0.6], np.float64), name)
    if clip is not None:
        p.clip_n, p.clip_off = np.asarray(clip[0], np.float64) / np.linalg.norm(clip[0]), float(clip[1])
    p.only_back = only_back
    return p


def _finish(family, seed, parts, rng, meta) -> ProcAsset:
    accent = _col(rng, 0.0)
    a = ProcAsset(family, seed, parts, accent, _complement(accent), rng.uniform(-0.15, 0.15, 2) + np.array([0.0, 0.1]),
                  float(rng.uniform(0.07, 0.14)), float(rng.uniform(4.0, 7.0)), float(rng.uniform(0, 2.0)), meta)
    a._normalise()
    return a


def _build_prop(rng, seed) -> ProcAsset:
    n = int(rng.integers(2, 5))
    parts = []
    for i in range(n):
        kind = ["sphere", "box", "rbox", "cylinder", "capsule", "torus", "cone"][int(rng.integers(7))]
        s = float(rng.uniform(0.18, 0.42))
        size = {"sphere": [s], "box": rng.uniform(0.6, 1.1, 3) * s, "rbox": list(rng.uniform(0.6, 1.1, 3) * s) + [0.3 * s],
                "cylinder": [s * rng.uniform(0.5, 0.9), s * rng.uniform(0.6, 1.4)], "capsule": [s * rng.uniform(0.3, 0.6), s * rng.uniform(0.5, 1.3)],
                "torus": [s * 0.8, s * rng.uniform(0.2, 0.4)], "cone": [s * rng.uniform(0.6, 1.0), s * rng.uniform(0.7, 1.4)]}[kind]
        t = rng.uniform(-0.4, 0.4, 3) * (1.0 if i else 0.25)
        rot = tuple(rng.uniform(-0.6, 0.6, 3)) if kind not in ("sphere",) else (0, 0, 0)
        parts.append(_mk(kind, size, t, rot, _col(rng), f"p{i}"))
    return _finish("prop", seed, parts, rng, {"n_parts": n})


def _build_blade(rng, seed) -> ProcAsset:
    steel, wood, trim = _PALETTE[7] * rng.uniform(0.85, 1.0), _PALETTE[9] * rng.uniform(0.8, 1.1), _col(rng)
    parts = []
    curved = bool(rng.random() < 0.45)
    L = float(rng.uniform(0.9, 1.5))                                    # overall length of the shaft
    parts.append(_mk("capsule", [0.045, L / 2], [0, 0, 0], colour=wood, name="shaft"))
    parts.append(_mk("sphere", [0.07], [0, -L / 2 - 0.02, 0], colour=trim, name="pommel"))
    if curved:                                                           # scythe: a torus arc on top, hooked to the front
        Rr = float(rng.uniform(0.28, 0.42))
        # a ring whose axis is world z; keep its lower half (local z >= 0 is world y <= centre) -> a hooked, smile-shaped blade
        parts.append(_mk("torus", [Rr, 0.03], [Rr, L / 2 - 0.02, 0.0], (np.pi / 2, 0, 0), steel, "blade", clip=([0.0, 0.0, -1.0], 0.0)))
    else:                                                                # sword
        w = float(rng.uniform(0.05, 0.1))
        parts.append(_mk("rbox", [w, L * 0.38, 0.016, 0.012], [0, L / 2 + L * 0.3, 0.0], colour=steel, name="blade"))
        parts.append(_mk("rbox", [w * 3.2, 0.03, 0.035, 0.015], [0, L / 2 - 0.02, 0.0], colour=trim, name="guard"))
    gems = int(rng.integers(0, 3))
    for g in range(gems):
        parts.append(_mk("sphere", [0.032], [0.0, -L / 4 + g * 0.16, 0.055], colour=_col(rng), name=f"gem{g}"))
    return _finish("blade", seed, parts, rng, {"curved": curved, "gems": gems, "length": L})


def _build_figure(rng, seed) -> ProcAsset:
    skin = _SKIN[int(rng.integers(len(_SKIN)))] + rng.normal(0, 0.02, 3)
    shirt, pants, cape_c, hair = _col(rng), _col(rng), _col(rng), _col(rng)
    tr = float(rng.uniform(0.15, 0.21))
    parts = [_mk("capsule", [tr, 0.2], [0, 0.08, 0], colour=shirt, name="torso"),
             _mk("sphere", [float(rng.uniform(0.12, 0.16))], [0, 0.52, 0.0], colour=skin, name="head")]
    ang = float(rng.uniform(-0.4, 1.9))                                  # right arm raised by `ang`, left arm hangs
    for sgn, a in ((1.0, 0.12), (-1.0, ang)):
        c = np.array([sgn * (tr + 0.07), 0.2, 0.0])
        d = np.array([sgn * np.sin(a), -np.cos(a), 0.0])
        parts.append(_mk("capsule", [0.055, 0.17], c + d * 0.17, (0, 0, np.arctan2(-d[0], d[1])), skin if rng.random() < 0.3 else shirt, "arm"))
    for sgn in (1.0, -1.0):
        parts.append(_mk("capsule", [0.07, 0.2], [sgn * 0.1, -0.42, 0.0], colour=pants, name="leg"))
    hat = bool(rng.random() < 0.5)
    if hat:
        parts.append(_mk("cone", [0.17, float(rng.uniform(0.12, 0.25))], [0, 0.72, 0.0], colour=hair, name="hat"))
    cape = bool(rng.random() < 0.5)
    if cape:
        parts.append(_mk("rbox", [tr + 0.12, 0.34, 0.025, 0.01], [0, 0.0, -tr - 0.03], colour=cape_c, name="cape", only_back=True))
    return _finish("figure", seed, parts, rng, {"cape": cape, "hat": hat, "arm_angle": ang})


def _build_tower(rng, seed) -> ProcAsset:
    stone = _PALETTE[7] * rng.uniform(0.55, 0.9)
    roof = _col(rng)
    r0 = float(rng.uniform(0.26, 0.38))
    tiers = int(rng.integers(1, 4))
    parts, y = [], -0.6
    for i in range(tiers):
        h = float(rng.uniform(0.16, 0.3))
        r = r0 * (1.0 - 0.18 * i)
        parts.append(_mk("cylinder", [r, h], [0, y + h, 0], colour=stone * (1.0 - 0.06 * i), name=f"tier{i}"))
        y += 2 * h
    parts.append(_mk("cone", [r0 * 1.15, float(rng.uniform(0.2, 0.4))], [0, y + 0.25, 0], colour=roof, name="roof"))
    door = _mk("rbox", [0.08, 0.14, 0.05, 0.02], [0, -0.6 + 0.14, r0 - 0.01], colour=_PALETTE[11], name="door")
    parts.append(door)
    banner = bool(rng.random() < 0.5)
    if banner:
        parts.append(_mk("rbox", [0.02, 0.2, 0.12, 0.008], [0, -0.1, -r0 - 0.03], colour=_col(rng), name="banner", only_back=True))
    return _finish("tower", seed, parts, rng, {"tiers": tiers, "banner": banner})


_BUILD = {"prop": _build_prop, "blade": _build_blade, "figure": _build_figure, "tower": _build_tower}


def make_asset(family: Optional[str] = None, seed: int = 0) -> ProcAsset:
    rng = np.random.default_rng(int(seed) * 7919 + 13)
    if family is None:
        family = FAMILIES[int(rng.integers(len(FAMILIES)))]
    return _BUILD[family](rng, seed)
