"""Projective baking of analytic charts from the reference picture, with a depth test and hidden-side fills."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .camera import OrthoCamera, rasterize_ortho, render_view
from .layers import Chart, Layer


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(x - 0.5, 0, w - 1.001)
    y = np.clip(y - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    f = img.astype(np.float32)
    return ((f[y0, x0] * (1 - fx) + f[y0, x0 + 1] * fx) * (1 - fy) + (f[y0 + 1, x0] * (1 - fx) + f[y0 + 1, x0 + 1] * fx) * fy)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def reflect_fill(img: np.ndarray, sure: np.ndarray, cyclic: bool = False) -> np.ndarray:
    """Fill the texels that are not `sure` along each row by reflecting the content just beside the gap into it
    (from both ends, cross-faded): the pattern continues across the gap instead of smearing, and there is no seam
    where the gap meets the good texels.  Rows without any sure texel are left to the caller."""
    h, w = sure.shape
    idx = np.broadcast_to(np.arange(w)[None, :], (h, w))
    any_sure = sure.any(1)
    first = np.where(any_sure, np.argmax(sure, 1), 0)
    last = np.where(any_sure, w - 1 - np.argmax(sure[:, ::-1], 1), 0)
    li = np.maximum.accumulate(np.where(sure, idx, -10 * w), axis=1)
    ri = np.minimum.accumulate(np.where(sure, idx, 10 * w)[:, ::-1], axis=1)[:, ::-1]
    if cyclic:
        li = np.where(li < 0, (last - w)[:, None], li)
        ri = np.where(ri >= w, (first + w)[:, None], ri)
    else:
        li = np.where(li < 0, ri, li)
        ri = np.where(ri >= w, li, ri)
    rows = np.arange(h)[:, None]

    def pick(x):
        xi = np.mod(x, w) if cyclic else np.clip(x, 0, w - 1)
        return img[rows, xi]

    ml = 2 * li - idx                   # mirror image of the texel about the left boundary of the gap
    mr = 2 * ri - idx
    left = pick(ml)
    right = pick(mr)
    # a mirrored texel that is itself unsure (gap wider than the good side) falls back to the boundary texel
    left = np.where(((np.mod(ml, w) if cyclic else np.clip(ml, 0, w - 1)) >= 0)[..., None] & sure[rows, np.mod(ml, w) if cyclic else np.clip(ml, 0, w - 1)][..., None], left, pick(li))
    right = np.where(sure[rows, np.mod(mr, w) if cyclic else np.clip(mr, 0, w - 1)][..., None], right, pick(ri))
    dl = (idx - li).astype(np.float32)
    dr = (ri - idx).astype(np.float32)
    tot = np.maximum(dl + dr, 1.0)
    wl = smoothstep(dr / tot)[..., None]
    out = left * wl + right * (1 - wl)
    return np.where(any_sure[:, None, None], np.where(sure[..., None], img, out), img)


def _sample(chart: Chart, layer: Layer, sel: np.ndarray, cam: OrthoCamera, depth_ref: np.ndarray,
            src_filled: np.ndarray, alpha: np.ndarray, eps: float, grazing: float):
    """Rasterise the faces `sel` of one chart in UV space and look every texel up in the reference picture.
    -> out (h,w,3), visible (h,w) bool, conf (h,w) 0..1, covered (h,w) bool, facing (h,w).
    The picture position of every texel is left in ``_sample.last_xy`` as (px, py) maps (NaN where the chart is empty)."""
    w = max(4, int(round(chart.width * chart.scale)))
    h = max(4, int(round(chart.height * chart.scale)))
    faces = layer.faces[sel]
    uv = layer.corner_uv[sel].astype(np.float64) * chart.scale
    xy = uv.reshape(-1, 2)                                  # one "vertex" per corner so seams need no special care
    tri_idx = np.arange(len(sel) * 3).reshape(-1, 3)
    tid, bary, _ = rasterize_ortho(xy, np.zeros(len(xy)), tri_idx, h, w)
    hit = tid >= 0
    out = np.zeros((h, w, 3), np.float32)
    visible = np.zeros((h, w), bool)
    cmap = np.zeros((h, w), np.float32)
    fmap = np.zeros((h, w), np.float32)
    pxm = np.full((h, w), np.nan, np.float32)
    pym = np.full((h, w), np.nan, np.float32)
    _sample.last_xy = (pxm, pym)
    if not hit.any():
        return out, visible, cmap, hit, fmap
    t = tid[hit]
    b = bary[hit].astype(np.float64)
    V = layer.verts[faces[t]]                                  # (n,3,3)
    P = (V * b[..., None]).sum(1)
    N = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])         # flat (geometric) normals: robust at creases
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    px, py, tt = cam.project(P)
    inside = (px >= 0) & (px < cam.width) & (py >= 0) & (py < cam.height)
    ix = np.clip(px.astype(np.int64), 0, cam.width - 1)
    iy = np.clip(py.astype(np.int64), 0, cam.height - 1)
    zbuf = depth_ref[iy, ix]
    facing = N @ cam.to_camera
    vis = inside & np.isfinite(zbuf) & (tt >= zbuf - eps) & (facing > 0.0) & (alpha[iy, ix] >= 0.3)
    col = _bilinear(src_filled, np.clip(px, 0, cam.width), np.clip(py, 0, cam.height))
    conf = np.where(vis, smoothstep(facing / grazing), 0.0)
    hh, ww = np.nonzero(hit)
    out[hh, ww] = col
    visible[hh, ww] = vis & (conf > 0.0)
    cmap[hh, ww] = conf
    fmap[hh, ww] = np.where(inside, facing, 0.0)
    pxm[hh, ww] = px
    pym[hh, ww] = py
    return out, visible, cmap, hit, fmap


def bake_param(chart: Chart, layer: Layer, vnormals: np.ndarray, cam: OrthoCamera, depth_ref: np.ndarray,
               src_filled: np.ndarray, alpha: np.ndarray, eps: float = 2.5, grazing: float = 0.18,
               blend_px: float = 24.0, trust_min: float = 0.04, uncertain: float = 0.35) -> np.ndarray:
    """-> (h, w, 3) uint8 for the chart at its packed scale.  Visible texels are the reference picture sampled where the
    depth test says the reference camera sees them; the rest is filled by `chart.fill`:
        through   project the texel into the picture ignoring occlusion (a sphere's far side = the mirrored picture)
        rot180    a body of revolution seen from one side: copy the texel half a turn around (u + width / 2)
        planar180 a planar disc: point-reflect through the chart centre
        fold2     the chart is half a turn wide and serves two opposite halves of a revolve: whichever half the picture
                  sees best supplies the texel (so the hidden half of the object is a turned copy of the visible half)
        nearest   the nearest visible texel
    """
    from scipy import ndimage as ndi
    w = max(4, int(round(chart.width * chart.scale)))
    h = max(4, int(round(chart.height * chart.scale)))
    if chart.fill == "mesh":                                       # a supplied mesh: colours solved on the surface
        from .meshfill import bake_mesh_chart
        return bake_mesh_chart(chart, layer, cam, depth_ref, src_filled, alpha, eps=eps, **chart.params)
    sel_all = np.flatnonzero(layer.face_chart == chart.cid)
    if len(sel_all) == 0:
        return np.zeros((h, w, 3), np.uint8)
    fill = chart.fill
    if fill == "fold2":
        side = layer.sides()[sel_all]
        parts = [_sample(chart, layer, sel_all[side == k], cam, depth_ref, src_filled, alpha, eps, grazing)
                 for k in (0, 1)]
        (oA, vA, cA, hA, _), (oB, vB, cB, hB, _) = parts
        pickB = cB > cA
        out = np.where(pickB[..., None], oB, oA)
        cmap = np.where(pickB, cB, cA)
        visible = vA | vB
        covered = hA | hB
    else:
        out, visible, cmap, covered, facing_map = _sample(chart, layer, sel_all, cam, depth_ref, src_filled, alpha,
                                                          eps, grazing)
        if not covered.any():
            return out.astype(np.uint8)
    valid = covered.copy()
    # ---- fill the texels the picture cannot see
    if fill == "through":
        # every texel takes the picture colour at its own projection: the far side of a sphere is the mirrored picture.
        # Texels at a grazing angle (the limb, front or back) would smear the last two or three picture pixels over a
        # wide band, so they are untrusted and extended from the nearest trusted texel instead.
        tm = float(chart.params.get("trust_min", trust_min))
        if chart.params.get("reflect"):
            # texels the picture sees almost edge-on (a band along the limb) are stretched and noisy: rebuild them by
            # reflecting the good texels on both sides of the band, cross-faded - the surface pattern continues
            af = np.abs(facing_map)
            sure = covered & (af >= 1.5 * tm)
            if sure.any() and not sure.all():
                filled = reflect_fill(out, sure, cyclic=True)
                rows_ok = sure.any(1)
                _, (iy2, ix2) = ndi.distance_transform_edt(~sure, return_indices=True)
                filled = np.where(rows_ok[:, None, None], filled, out[iy2, ix2])
                a = smoothstep((af - tm) / max(0.5 * tm, 1e-6))[..., None]
                out = np.where(sure[..., None], out, out * a + filled * (1 - a))
        else:
            trust = covered & (np.abs(facing_map) >= tm)
            if trust.any() and not trust.all():
                _, (iy2, ix2) = ndi.distance_transform_edt(~trust, return_indices=True)
                out = np.where(trust[..., None], out, out[iy2, ix2])
    elif fill in ("rot180", "planar180"):
        if fill == "rot180":
            sh = w // 2
            src_c = np.roll(out, sh, axis=1)
            src_v = np.roll(visible, sh, axis=1)
        else:
            src_c = out[::-1, ::-1]
            src_v = visible[::-1, ::-1]
        # near the visible/hidden border fade between "extend the nearest visible texel" and "the rotated copy"
        d, (iy2, ix2) = ndi.distance_transform_edt(~visible, return_indices=True)
        near = out[iy2, ix2]
        k = smoothstep(d / max(blend_px * chart.scale, 1.0))[..., None]
        hidden_col = np.where(src_v[..., None], near * (1 - k) + src_c * k, near)
        out = np.where(visible[..., None], out, hidden_col)
        # grazing texels: mix the picture with the rotated copy so the terminator does not smear
        mix = (cmap < 1.0) & visible & src_v
        a = cmap[..., None]
        out = np.where(mix[..., None], out * a + src_c * (1 - a), out)
    else:                                                          # nearest | fold2
        sure = visible & (cmap >= uncertain)                       # texels the picture sees well
        if sure.any() and not sure.all():
            if fill == "fold2":
                near = reflect_fill(out, sure, cyclic=bool(chart.params.get("wrap_u")))
                rows_ok = sure.any(1)
                _, (iy2, ix2) = ndi.distance_transform_edt(~sure, return_indices=True)
                near = np.where(rows_ok[:, None, None], near, out[iy2, ix2])
            else:
                _, (iy2, ix2) = ndi.distance_transform_edt(~sure, return_indices=True)
                near = out[iy2, ix2]
            a = np.where(visible, smoothstep(cmap / uncertain), 0.0)[..., None]   # grazing texels: blend towards the neighbours
            out = out * a + near * (1 - a)
    # gutters: everything not covered by a face copies the nearest covered texel (the packer adds a padded copy too)
    if not valid.all():
        _, (iy3, ix3) = ndi.distance_transform_edt(~valid, return_indices=True)
        out = out[iy3, ix3]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def reference_depth(layers, cam: OrthoCamera, scale: float = 1.0) -> np.ndarray:
    from .layers import merge_layers
    m = merge_layers(layers)
    _, _, t = render_view(cam, m.verts, m.faces)
    return t
