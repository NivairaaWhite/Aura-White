"""Colouring a mesh the picture only half sees (the 'mesh' chart fill).

Every texel of the atlas is a point on the surface.  What the reference camera sees (depth test + facing test) is the
picture itself - that is the *lock*, pixel exact.  The rest (the sides, the back, anything occluded) is not mirrored or
stretched: the colour is solved **on the mesh surface** as a harmonic (Laplace) interpolation, with the colours of the
visible vertices - and of the vertices a back-view picture shows, when you supply one - as boundary data.  A back of
hair is hair-coloured, a cloak's back is cloak-coloured, a rock's far side is rock, and colours run smoothly round the
silhouette from the front to the back instead of being cut off.  A small amount of picture detail can be added back
(`detail`) so the unseen sides are not perfectly smooth.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .bake import _bilinear, smoothstep
from .camera import OrthoCamera, rasterize_ortho, render_view
from .layers import Chart, Layer


def _vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _blur_rgb(img: np.ndarray, sigma: float) -> np.ndarray:
    from scipy import ndimage as ndi
    return np.stack([ndi.gaussian_filter(img[..., c].astype(np.float32), sigma) for c in range(3)], -1)


def harmonic_colours(v: np.ndarray, f: np.ndarray, fixed: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """Per-vertex colours (n,3): fixed vertices keep `colour`, the others solve  L x = 0  on the mesh graph with
    inverse-edge-length weights (so the result follows the surface, not the 3-D neighbourhood)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import splu
    n = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    w = 1.0 / (np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1) + 1e-6)
    w = w / w.mean()
    W = sparse.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    fixed = fixed.copy()
    colour = colour.astype(np.float64).copy()
    # a piece of surface with no visible vertex at all borrows the colour of the nearest visible vertex
    ncomp, lab = connected_components(W, directed=False)
    has = np.zeros(ncomp, bool)
    has[lab[fixed]] = True
    if not has.all() and fixed.any():
        from scipy.spatial import cKDTree
        tree = cKDTree(v[fixed])
        fx_idx = np.flatnonzero(fixed)
        for c in np.flatnonzero(~has):
            members = np.flatnonzero(lab == c)
            cen = v[members].mean(0)
            _, j = tree.query(cen)
            seed = members[np.argmin(np.linalg.norm(v[members] - cen, axis=1))]
            fixed[seed] = True
            colour[seed] = colour[fx_idx[j]]
    if not fixed.any():
        return np.full((n, 3), 128.0)
    free = ~fixed
    out = colour.copy()
    if not free.any():
        return out
    d = np.asarray(W.sum(1)).ravel()
    L = sparse.diags(d) - W
    ui, fi = np.flatnonzero(free), np.flatnonzero(fixed)
    Luu = L[ui][:, ui].tocsc()
    rhs = -(L[ui][:, fi] @ colour[fi])
    try:
        lu = splu(Luu + sparse.identity(len(ui), format="csc") * 1e-9)
        out[ui] = lu.solve(rhs)
    except Exception:                       # a sparse LU that runs out of memory must not stop the build: iterate instead
        from scipy.sparse.linalg import cg
        for c in range(3):
            x, _ = cg(Luu, rhs[:, c], x0=np.full(len(ui), colour[fi, c].mean()), rtol=1e-4, maxiter=4000)
            out[ui, c] = x
    return np.clip(out, 0.0, 255.0)


class SourceColour:
    """The colours a supplied mesh carries itself (its texture or vertex colours), as a nearest-neighbour field in the
    picture's world.  Generators such as Hunyuan3D, TRELLIS or Meshy invent a plausible back and sides; these are the
    colours used there, tone-matched to the picture where the picture can see the surface."""

    def __init__(self, pts: np.ndarray, rgb: np.ndarray):
        from scipy.spatial import cKDTree
        self.tree = cKDTree(np.asarray(pts, np.float64))
        self.rgb = np.asarray(rgb, np.float32)
        self.gain = np.ones(3, np.float32)
        self.bias = np.zeros(3, np.float32)

    def raw(self, P: np.ndarray, k: int = 3) -> np.ndarray:
        d, j = self.tree.query(P, k=k, workers=-1)
        if k == 1:
            return self.rgb[j]
        w = 1.0 / (d + 1e-6 + 0.25 * d[:, :1])
        w /= w.sum(1, keepdims=True)
        return (self.rgb[j] * w[..., None]).sum(1)

    def fit_tone(self, P: np.ndarray, picture: np.ndarray) -> None:
        """Per-channel gain/bias that bring the mesh's colours to the picture's, from surface points the picture sees."""
        if len(P) < 50:
            return
        x = self.raw(P, k=4).astype(np.float64)
        y = picture.astype(np.float64)
        keep = np.ones(len(x), bool)
        for c in range(3):
            a, b = 1.0, 0.0
            keep = np.ones(len(x), bool)
            for _ in range(3):
                if keep.sum() < 30 or np.ptp(x[keep, c]) < 8:
                    break
                A = np.stack([x[keep, c], np.ones(keep.sum())], 1)
                a, b = np.linalg.lstsq(A, y[keep, c], rcond=None)[0]
                r = np.abs(y[:, c] - (a * x[:, c] + b))
                keep = r <= max(6.0, 2.5 * 1.4826 * np.median(r[keep]))
            self.gain[c] = float(np.clip(a, 0.6, 1.6))
            self.bias[c] = float(np.clip(b, -60.0, 60.0))

    def __call__(self, P: np.ndarray, k: int = 3) -> np.ndarray:
        return np.clip(self.raw(P, k) * self.gain + self.bias, 0.0, 255.0)


def _vertex_data(v, f, vn, cam: OrthoCamera, depth_ref, src_blur, alpha, eps, grazing, flip: bool):
    """Visibility confidence and picture colour of every vertex for the camera (or its half-turned twin when `flip`)."""
    if flip:
        v = v * np.array([-1.0, 1.0, -1.0])
        vn = vn * np.array([-1.0, 1.0, -1.0])
    px, py, tt = cam.project(v)
    H, W = depth_ref.shape
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    ix, iy = np.clip(px.astype(np.int64), 0, W - 1), np.clip(py.astype(np.int64), 0, H - 1)
    z = depth_ref[iy, ix]
    facing = vn @ cam.to_camera
    ok = inside & np.isfinite(z) & (tt >= z - eps) & (facing > 0)
    conf = np.where(ok, smoothstep(facing / grazing), 0.0)
    col = _bilinear(src_blur, np.clip(px, 0, W), np.clip(py, 0, H))
    return conf, col, px, py


def bake_mesh_chart(chart: Chart, layer: Layer, cam: OrthoCamera, depth_ref: np.ndarray, src_filled: np.ndarray,
                    alpha: np.ndarray, eps: float = 2.5, grazing: float = 0.22, style: str = "soft", detail: float = 0.3,
                    back_img: Optional[np.ndarray] = None, src_col: Optional[SourceColour] = None,
                    skin_start: Optional[int] = None, axis_x: float = 0.0, **_) -> np.ndarray:
    """-> (h, w, 3) uint8 tile of the chart at its packed scale (see the module docstring)."""
    from scipy import ndimage as ndi
    w = max(4, int(round(chart.width * chart.scale)))
    h = max(4, int(round(chart.height * chart.scale)))
    v = layer.verts.astype(np.float64)
    f = layer.faces
    vn = _vertex_normals(v, f)
    H, W = alpha.shape
    c_cam = cam.to_camera

    # ---------------------------------------------------------------- vertex colours (the surface-diffusion part)
    px0, py0, _ = cam.project(v)
    e = np.concatenate([np.linalg.norm(np.stack([px0[f[:, a]] - px0[f[:, b]], py0[f[:, a]] - py0[f[:, b]]], 1), axis=1)
                        for a, b in ((0, 1), (1, 2), (2, 0))])
    med_edge = float(np.median(e[e > 0])) if (e > 0).any() else 2.0
    sigma_v = float(np.clip(0.6 * med_edge, 1.0, 14.0))
    src_blur = _blur_rgb(src_filled, sigma_v)
    conf_f, col_f, _, _ = _vertex_data(v, f, vn, cam, depth_ref, src_blur, alpha, eps, grazing, False)
    conf_b = np.zeros(len(v))
    col_b = col_f
    depth_back = None
    if back_img is not None:
        # a back-view picture, registered with the front framing (mirrored); the half-turned mesh seen by the same camera
        vb = v * np.array([-1.0, 1.0, -1.0])
        _, _, tb = render_view(cam, vb, f)
        depth_back = tb
        back_blur = _blur_rgb(back_img, sigma_v)
        conf_b, _, _pxb, pyb = _vertex_data(v, f, vn, cam, depth_back, back_blur, alpha, eps, grazing, True)
        # the picture is registered with the front framing (mirrored): column of the front projection, row of the twin's
        col_b = _bilinear(back_blur, np.clip(px0, 0, W), np.clip(pyb, 0, H))
    echo = style == "echo" and back_img is None and src_col is None
    if echo:
        # 'echo': what the picture shows on the near side is reflected through the object's mid-plane (front <-> back, same
        # column) and worn by the far side - the far side of a shaft carries the near side's look, a far wall the near wall's -
        # and the surface diffusion blends the two over the flanks, so nothing is stretched sideways
        flip = np.array([1.0, 1.0, -1.0])
        t_all = cam.project(v)[2]
        eps_e = max(eps, 0.02 * float(t_all.max() - t_all.min()))     # the mid-plane is only roughly the middle of a lopsided shape
        vt = np.stack([v[:, 0], v[:, 1], -v[:, 2]], 1)
        pxt, pyt, ttt = cam.project(vt)
        ins = (pxt >= 0) & (pxt < W) & (pyt >= 0) & (pyt < H)
        ixt, iyt = np.clip(pxt.astype(np.int64), 0, W - 1), np.clip(pyt.astype(np.int64), 0, H - 1)
        zt = depth_ref[iyt, ixt]
        ft = (vn * flip) @ c_cam
        okt = ins & np.isfinite(zt) & (ttt >= zt - eps_e) & (ft > 0) & (alpha[iyt, ixt] >= 0.3)
        conf_b = np.where(okt, smoothstep(ft / grazing), 0.0)
        col_b = _bilinear(src_blur, np.clip(pxt, 0, W), np.clip(pyt, 0, H))
    if src_col is not None:
        vis_v = conf_f >= 0.5
        if vis_v.any():
            src_col.fit_tone(v[vis_v], col_f[vis_v])
    use_b = conf_b > conf_f
    conf_v = np.where(use_b, conf_b, conf_f)
    col_v = np.where(use_b[:, None], col_b, col_f)
    fixed = conf_v >= 0.5
    if style == "mirror":
        solved = col_v
    else:
        solved = harmonic_colours(v, f, fixed, col_v)

    # ---------------------------------------------------------------- the texels
    own = np.flatnonzero(layer.face_chart == chart.cid)           # faces of other charts (the picture's own front chart) are not baked here
    uv = layer.corner_uv[own].astype(np.float64) * chart.scale
    xy = uv.reshape(-1, 2)
    tri_idx = np.arange(len(own) * 3).reshape(-1, 3)
    tid, bary, _ = rasterize_ortho(xy, np.zeros(len(xy)), tri_idx, h, w)
    tid = np.where(tid >= 0, own[np.clip(tid, 0, max(len(own) - 1, 0))] if len(own) else tid, -1)
    hit = tid >= 0
    out = np.zeros((h, w, 3), np.float32)
    if not hit.any():
        return out.astype(np.uint8)
    hh, ww = np.nonzero(hit)
    src_detail = _blur_rgb(src_filled, max(2.0, 2.0 * sigma_v)) if detail > 0 else None
    CH = 400_000
    for s0 in range(0, len(hh), CH):
        sl = slice(s0, s0 + CH)
        t = tid[hh[sl], ww[sl]]
        b = bary[hh[sl], ww[sl]].astype(np.float64)
        tri = f[t]
        P = (v[tri] * b[..., None]).sum(1)
        Nf = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
        Nf /= np.maximum(np.linalg.norm(Nf, axis=1, keepdims=True), 1e-12)
        Ns = (vn[tri] * b[..., None]).sum(1)
        Ns /= np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-12)
        px, py, tt = cam.project(P)
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        ix, iy = np.clip(px.astype(np.int64), 0, W - 1), np.clip(py.astype(np.int64), 0, H - 1)
        z = depth_ref[iy, ix]
        vis = inside & np.isfinite(z) & (tt >= z - eps) & ((Nf @ c_cam) > 0) & (alpha[iy, ix] >= 0.3)
        wf = np.where(vis, smoothstep((Ns @ c_cam) / grazing), 0.0)[:, None]
        c_front = _bilinear(src_filled, np.clip(px, 0, W), np.clip(py, 0, H))
        c_hid = (solved[tri] * b[..., None]).sum(1)
        if src_col is not None:                 # the mesh's own colours for everything the picture does not show
            c_s = src_col(P)
            if skin_start is not None:           # picture-relief patches have no counterpart in the source mesh
                c_s = np.where((t >= skin_start)[:, None], c_hid, c_s)
            c_hid = c_s
        if detail > 0 and style != "mirror":
            thru = c_front
            c_hid = c_hid + detail * (thru - _bilinear(src_detail, np.clip(px, 0, W), np.clip(py, 0, H)))
        if style == "mirror":
            c_hid = c_front
        if back_img is not None and depth_back is not None:
            Pb = P * np.array([-1.0, 1.0, -1.0])
            pxb2, pyb2, ttb = cam.project(Pb)
            insb = (pxb2 >= 0) & (pxb2 < W) & (pyb2 >= 0) & (pyb2 < H)
            ixb, iyb = np.clip(pxb2.astype(np.int64), 0, W - 1), np.clip(pyb2.astype(np.int64), 0, H - 1)
            zb = depth_back[iyb, ixb]
            Nb = Nf * np.array([-1.0, 1.0, -1.0])
            visb = insb & np.isfinite(zb) & (ttb >= zb - eps) & ((Nb @ c_cam) > 0)
            Nsb = Ns * np.array([-1.0, 1.0, -1.0])
            wb = np.where(visb, smoothstep((Nsb @ c_cam) / grazing), 0.0)[:, None]
            c_back = _bilinear(back_img, np.clip(px, 0, W), np.clip(pyb2, 0, H))
            c_hid = wb * c_back + (1.0 - wb) * c_hid
        if echo:
            Pt = np.stack([P[:, 0], P[:, 1], -P[:, 2]], 1)
            pxe, pye, tte = cam.project(Pt)
            ine = (pxe >= 0) & (pxe < W) & (pye >= 0) & (pye < H)
            ixe, iye = np.clip(pxe.astype(np.int64), 0, W - 1), np.clip(pye.astype(np.int64), 0, H - 1)
            ze = depth_ref[iye, ixe]
            vise = ine & np.isfinite(ze) & (tte >= ze - eps_e) & (((Nf * np.array([1.0, 1.0, -1.0])) @ c_cam) > 0) & \
                (alpha[iye, ixe] >= 0.3)
            we = np.where(vise, smoothstep(((Ns * np.array([1.0, 1.0, -1.0])) @ c_cam) / grazing), 0.0)[:, None]
            c_echo = _bilinear(src_filled, np.clip(pxe, 0, W), np.clip(pye, 0, H))
            c_hid = we * c_echo + (1.0 - we) * c_hid
        out[hh[sl], ww[sl]] = wf * c_front + (1.0 - wf) * c_hid
    # gutters: texels no face covers copy the nearest covered texel (mip-map and bilinear safe)
    if not hit.all():
        _, (iy3, ix3) = ndi.distance_transform_edt(~hit, return_indices=True)
        out = out[iy3, ix3]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
