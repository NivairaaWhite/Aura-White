"""Cameras in the canonical Aura White frame (identical to TripoSR / Zero123++ / InstantMesh):

    z is up, the input photo is taken from +x (azimuth 0), image-right is +y.

Cameras are OpenGL style (looks down -z, x right, y up).  Intrinsics are resolution independent:
``f`` = 1 / tan(fov / 2) and the principal point is an offset (cx, cy) in NDC units, so a camera
works unchanged for a 320 px generated view and a 4096 px reference picture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

# Zero123++ v1.2 renders 6 views at these azimuths (relative to the input photo) and absolute elevations.
ZERO123PP_AZIMUTHS = (30.0, 90.0, 150.0, 210.0, 270.0, 330.0)
ZERO123PP_ELEVATIONS = (20.0, -10.0, 20.0, -10.0, 20.0, -10.0)
ZERO123PP_FOV = 30.0
ZERO123PP_RADIUS = 4.0  # camera distance when the object fits in a +-1.05 cube (InstantMesh convention)


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World -> camera matrix (4x4, OpenGL: camera looks along -z)."""
    eye = np.asarray(eye, dtype=np.float64)
    z = eye - np.asarray(target, dtype=np.float64)
    z /= max(np.linalg.norm(z), 1e-12)
    x = np.cross(np.asarray(up, dtype=np.float64), z)
    nx = np.linalg.norm(x)
    if nx < 1e-9:  # looking straight along `up`
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return np.linalg.inv(c2w)


@dataclass(frozen=True)
class Camera:
    w2c: np.ndarray            # (4,4) float64
    f: float = 1.0 / math.tan(math.radians(30.0) / 2)   # focal length in NDC units (see module doc)
    cx: float = 0.0            # principal point offset, NDC (+x right)
    cy: float = 0.0            # principal point offset, NDC (+y up)

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c)

    @property
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def fov(self) -> float:
        return math.degrees(2.0 * math.atan(1.0 / self.f))

    def with_intrinsics(self, f=None, cx=None, cy=None) -> "Camera":
        return Camera(self.w2c, self.f if f is None else f, self.cx if cx is None else cx,
                      self.cy if cy is None else cy)


def fov_to_f(fov_deg: float) -> float:
    return 1.0 / math.tan(math.radians(fov_deg) / 2.0)


def orbit_camera(azimuth: float, elevation: float, distance: float = ZERO123PP_RADIUS, fov: float = ZERO123PP_FOV,
                 cx: float = 0.0, cy: float = 0.0, target: Sequence[float] = (0.0, 0.0, 0.0)) -> Camera:
    """Camera on a sphere around `target`. Azimuth 0 = +x (the photo), 90 = +y; elevation 90 = straight above."""
    az, el = math.radians(azimuth), math.radians(float(np.clip(elevation, -89.0, 89.0)))
    tgt = np.asarray(target, dtype=np.float64)
    eye = tgt + distance * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return Camera(look_at(eye, tgt), fov_to_f(fov), cx, cy)


def zero123pp_cameras(azimuth0: float = 0.0, radius: float = ZERO123PP_RADIUS, fov: float = ZERO123PP_FOV):
    """The six Zero123++ v1.2 output cameras (row-major order of the 2x3 image grid)."""
    return [orbit_camera(azimuth0 + a, e, radius, fov) for a, e in zip(ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS)]


def normalize_mesh(verts: np.ndarray, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Centre on the bounding-box centre and scale so the farthest vertex is at `radius`.
    Returns (verts, centre, scale) with  new = (old - centre) * scale."""
    v = np.asarray(verts, dtype=np.float64)
    centre = (v.max(0) + v.min(0)) / 2.0
    scale = radius / max(float(np.linalg.norm(v - centre, axis=1).max()), 1e-12)
    return ((v - centre) * scale).astype(np.float32), centre, scale


def transform_camera(cam: Camera, centre: np.ndarray, scale: float) -> Camera:
    """Express a camera given in the original mesh frame in the frame  p' = (p - centre) * scale."""
    w2c = np.array(cam.w2c, dtype=np.float64)
    R, t = w2c[:3, :3], w2c[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = scale * (R @ np.asarray(centre, dtype=np.float64) + t)
    return Camera(out, cam.f, cam.cx, cam.cy)
