"""Synthetic scene used by the texturing tests: an asymmetric object with a known 3D colour field, plus fake
back-ends that behave like Hunyuan3D / Zero123++ but return ground truth."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from aura_white.backends.base import GeometryResult, MultiViewResult
from aura_white.raster import interpolate, orbit_camera
from aura_white.texture.register import make_camera


def make_object(sub: int = 3):
    body = trimesh.creation.icosphere(sub)
    body.apply_scale([0.5, 0.35, 0.8])
    tor = trimesh.creation.torus(0.45, 0.07, major_sections=48, minor_sections=14)
    tor.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    tor.apply_translation([0.1, 0.0, 0.75])
    obj = trimesh.util.concatenate([body, tor])
    v = np.asarray(obj.vertices, np.float32)
    f = np.asarray(obj.faces, np.int64)
    v = v - (v.max(0) + v.min(0)) / 2
    v = v / np.linalg.norm(v, axis=1).max()
    return v.astype(np.float32), f


def gt_color(P: np.ndarray) -> np.ndarray:
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    chk = (np.floor(x * 10) + np.floor(y * 10) + np.floor(z * 10)) % 2
    r = 0.55 + 0.4 * np.sin(6 * z + 1.0)
    g = 0.5 + 0.45 * np.sin(5 * x + 2 * y)
    b = 0.5 + 0.45 * np.cos(7 * y - 3 * z)
    c = np.stack([r, g, b], -1) * (0.78 + 0.22 * chk[..., None])
    return np.clip(c, 0, 1).astype(np.float32)


def render_gt(raster, v, f, cam, size, bg=0.5, blur=0):
    Vt, Ft = torch.as_tensor(v), torch.as_tensor(f)
    out = raster.rasterize_camera(Vt, Ft, cam, size)
    P = interpolate(Vt, Ft, out).numpy()
    img = gt_color(P)
    m = out.mask.numpy()
    if blur:
        x = torch.as_tensor(img).permute(2, 0, 1)[None]
        k = torch.as_tensor(np.hanning(2 * blur + 3)[1:-1], dtype=torch.float32)
        k = k / k.sum()
        x = F.conv2d(F.pad(x, (blur, blur, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).expand(3, 1, 1, -1).contiguous(), groups=3)
        x = F.conv2d(F.pad(x, (0, 0, blur, blur), mode="replicate"), k.view(1, 1, -1, 1).expand(3, 1, -1, 1).contiguous(), groups=3)
        img = x[0].permute(1, 2, 0).numpy()
    return img * m[..., None] + (1 - m[..., None]) * bg, m


def psnr(a, b, mask) -> float:
    d = (a - b)[mask]
    return 10 * math.log10(1.0 / max(float((d ** 2).mean()), 1e-12))


def make_artwork(raster, v, f, az=4.0, el=10.0, size=1024) -> Image.Image:
    """RGBA 'concept art': the object under an unknown camera, transparent background."""
    cam = make_camera(dict(az=az, el=el, logd=math.log(9.0), logf=math.log(1 / math.tan(math.radians(14) / 2)), cx=0.0, cy=0.0))
    img, m = render_gt(raster, v, f, cam, size)
    rgba = np.concatenate([img, m[..., None].astype(np.float32)], -1)
    return Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")


class FakeGeometry:
    name = "fake-geometry"

    def __init__(self, v, f, absolute=False, views=None):
        self.v, self.f, self.absolute, self.views = v, f, absolute, views

    def generate(self, image, **kw):
        return GeometryResult(self.v.copy(), self.f.copy(), absolute_scale=self.absolute, views=self.views)


class FakeMultiView:
    name = "fake-zero123"

    def __init__(self, raster, v, f, az0=4.0, blur=2, fov=27.0, dist=4.0):
        self.raster, self.v, self.f, self.az0, self.blur, self.fov, self.dist = raster, v, f, az0, blur, fov, dist
        self.unloaded = False

    def generate(self, image, **kw):
        from aura_white.raster.camera import ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS

        imgs = []
        for a, e in zip(ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS):
            cam = orbit_camera(self.az0 + a, e, self.dist, self.fov)
            img, _ = render_gt(self.raster, self.v, self.f, cam, 320, bg=0.5, blur=self.blur)
            imgs.append(img.astype(np.float32))
        return MultiViewResult(imgs, None, "gray")

    def unload(self):
        self.unloaded = True


def make_asymmetric_object(sub: int = 3):
    """make_object plus a spike on one side, so front/back and axis conventions are distinguishable."""
    v, f = make_object(sub)
    body = trimesh.Trimesh(v, f, process=False)
    spike = trimesh.creation.cone(0.12, 0.9)
    spike.apply_translation([0.0, 0.55, -0.2])
    obj = trimesh.util.concatenate([body, spike])
    v2 = np.asarray(obj.vertices, np.float32)
    v2 = v2 - (v2.max(0) + v2.min(0)) / 2
    return (v2 / np.linalg.norm(v2, axis=1).max()).astype(np.float32), np.asarray(obj.faces, np.int64)


def chamfer(a: np.ndarray, b: np.ndarray, n: int = 2000) -> float:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    a = a[rng.choice(len(a), min(n, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(n, len(b)), replace=False)]
    return float(0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()))
