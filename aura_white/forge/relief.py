"""Relief geometry back-end: a GPU-free, silhouette-exact reconstruction for thin / flat subjects (weapons, wings,
emblems, sheets, cut-out props, flat-shaded characters used as billboards).

The silhouette (holes included) is the artwork's own outline at sub-pixel accuracy; thin structures become round rods,
wide ones flat plates with a bevel; hairline chains and cords are detected and built as real interlocking links / tubes.
Because the geometry is derived from the picture in the picture's own camera, registration and texturing are exact
to the pixel.  It cannot invent a volumetric back for a character - use a neural back-end (pixal3d / trellis2 /
hunyuan3d) for that; `aura-white studio` can combine the two through residual completion."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..backends.base import GeometryResult, Progress
from . import morph, patch, wires as wr


def _medial_radius(mask: np.ndarray, dt: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    sk = morph.thin(mask)
    if not sk.any():
        return dt
    iy, ix = morph.nearest_index(sk)
    return np.maximum(ndi.median_filter(dt[iy, ix], size=5), dt)


class ReliefGeometry:
    name = "relief"

    def __init__(self, device: str = "auto", plate: float = 0.0045, rod: float = 0.013, wires: bool = True, **_):
        self.plate, self.rod, self.wires = plate, rod, wires

    def status(self):
        try:
            import scipy  # noqa: F401
        except Exception:
            return False, "needs SciPy (pip install scipy)"
        return True, "built in (no GPU)"

    def generate(self, cutout: Image.Image, seed: int = 0, progress: Progress = None, reference=None,
                 **_) -> GeometryResult:
        say = progress or (lambda *_: None)
        src = reference if reference is not None else cutout        # full-resolution artwork when available
        if max(src.size) > 2048:
            src = src.copy()
            src.thumbnail((2048, 2048), Image.LANCZOS)
        arr = np.asarray(src.convert("RGBA"))
        rgb, alpha = arr[..., :3], arr[..., 3].astype(np.float32) / 255.0
        mask = alpha >= 0.5
        H, W = mask.shape
        S = float(max(H, W))
        say("relief: measuring the silhouette")
        found, used = ([], np.zeros_like(mask))
        if self.wires:
            found, used = wr.find_wires(mask, rgb, bg=np.array([255.0, 255.0, 255.0]))
        body = mask & ~used
        from scipy import ndimage as ndi

        a = ndi.gaussian_filter(body.astype(np.float32), 0.7)
        dt = morph.edt(body)
        R = _medial_radius(body, dt)
        r_rod = self.rod * S
        w = patch.smoothstep((1.7 * r_rod - R) / r_rod)
        plate = patch.plate_profile(dt, self.plate * S, 0.012 * S)
        rod = patch.rod_profile(dt, np.maximum(R, 1.0))
        half = np.maximum(plate, w * rod).astype(np.float32)
        say("relief: meshing")
        rm = patch.build_relief(a, half, half, None, max_faces=160_000)
        if rm is None:
            raise RuntimeError("the artwork has no foreground to build a relief from")
        V = [rm.verts.astype(np.float64)]
        F = [rm.faces]
        base = len(rm.verts)
        n_chain = n_tube = 0
        for wire in found:
            z = 0.7 * self.plate * S
            P = np.stack([wire.pts[:, 0] + 0.5, wire.pts[:, 1] + 0.5, np.full(len(wire.pts), z)], 1)
            if wire.kind == "chain":
                v, f, n = wr.chain_mesh(P, wire.width, np.array([0.0, 0.0, -1.0]))
                n_chain += 1
            else:
                v, f = wr.tube_mesh(P, 0.45 * wire.width, taper=0.3)
                n_tube += 1
            if len(f):
                vol = float(np.sum(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]]))) / 6.0)
                V.append(v)
                F.append((f if vol >= 0 else f[:, [0, 2, 1]]) + base)
                base += len(v)
        v = np.concatenate(V, 0)
        f = np.concatenate(F, 0)
        # image space (x right, y down, z to the viewer) -> canonical (z up, viewer at +x, image-right = +y)
        can = np.stack([v[:, 2], v[:, 0] - W / 2.0, -(v[:, 1] - H / 2.0)], 1) / S
        vol = float(np.sum(np.einsum("ij,ij->i", can[f[:, 0]], np.cross(can[f[:, 1]], can[f[:, 2]]))) / 6.0)
        if vol < 0:
            f = f[:, [0, 2, 1]]
        return GeometryResult(can.astype(np.float32), f.astype(np.int64),
                              info={"backend": "relief", "faces": int(len(f)), "chains": n_chain, "tubes": n_tube})
