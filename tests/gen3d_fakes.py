"""Shared fakes for the gen3d tests: a shape generator that misbehaves on purpose, and picture renderers."""
from __future__ import annotations

import numpy as np
from PIL import Image

import prims
from aura_white.gen3d import shapegen as S
from aura_white.gen3d import views as V
from aura_white.gen3d.hardware import ExecPlan
from aura_white.gen3d.paint import camera_dir

TRUTH = prims.figure()
BG = (236, 236, 236)


def photo(az: float, el: float = 10.0, size: int = 384, scale: float = 128.0, tint=(200, 120, 90)) -> Image.Image:
    """The figure photographed on a plain background (flat-shaded, so it has real colour structure)."""
    cam = V.ViewCam(az, el, size, scale, 0.0, 0.25)
    m, _, tri, _ = V.render_maps(*TRUTH, cam)
    v, f = TRUTH
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    sh = 0.35 + 0.65 * np.clip(fn @ camera_dir(az, el), 0, 1)
    img = np.full((size, size, 3), BG, np.uint8)
    img[m] = np.clip(sh[tri[m]][:, None] * np.array(tint, float), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


class FakeShape(S.ShapeBackend):
    """Round 0 / candidate 0 returns a flat cut-out; everything later returns the real mesh in a z-up frame."""
    name, tier, multiview, stage = "fakegen", "universal", False, ""

    def __init__(self):
        self.calls = []

    def probe(self, hw, views):
        return S.Probe(True, "ready", ExecPlan("fake", "cpu"))

    def generate(self, call, views, plan, say):
        self.calls.append((call.round, call.index, call.guidance, [v.spec.label for v in views]))
        if call.round == 0 and call.index == 0:
            return prims.slab(1.1, 1.9, 0.05)
        Rz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        v, f = TRUTH
        return v @ Rz.T, f.copy()

    def close(self):
        pass


def install_fake(monkeypatch, fake=None):
    from aura_white.gen3d import engine
    fake = fake or FakeShape()
    monkeypatch.setattr(engine, "choose_backend", lambda views, hw, tier, allow_basic, say, force: S.Choice(fake, ExecPlan("fake", "cpu"), []))
    return fake
