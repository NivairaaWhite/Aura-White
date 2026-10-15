import math

import numpy as np
import torch
from synth import make_object, psnr, render_gt

from aura_white.mesh import vertex_normals
from aura_white.raster import get_raster, orbit_camera, zero123pp_cameras
from aura_white.texture import (ViewSource, align_view, bake_texture, estimate_foreground, find_orientation,
                                make_camera, register_reference, unwrap)
from aura_white.texture.preview import render_textured

R = get_raster("cpu")
V, F = make_object(2)


def asymmetric_object():
    import trimesh

    body = trimesh.Trimesh(V, F)
    spike = trimesh.creation.cone(0.12, 0.9)
    spike.apply_translation([0.0, 0.55, -0.2])          # a spike on one side: front and back silhouettes differ
    obj = trimesh.util.concatenate([body, spike])
    v = np.asarray(obj.vertices, np.float32)
    v = v - (v.max(0) + v.min(0)) / 2
    return (v / np.linalg.norm(v, axis=1).max()).astype(np.float32), np.asarray(obj.faces, np.int64)


def cam_params(az=9.0, el=17.0, d=7.0, fov=22.0, cx=0.03, cy=-0.02):
    return dict(az=az, el=el, logd=math.log(d), logf=math.log(1 / math.tan(math.radians(fov) / 2)), cx=cx, cy=cy)


def mask_for(t, size=256, v=V):
    return R.rasterize_camera(torch.tensor(v), torch.tensor(F), make_camera(t), size).mask.numpy()


def test_register_recovers_the_artwork_camera():
    reg = register_reference(R, V, F, mask_for(cam_params()), prior_az=0.0, prior_el=10.0)
    assert reg.iou > 0.95 and not reg.flipped
    assert abs(reg.params["az"] - 9.0) < 6 and abs(reg.params["cx"] - 0.03) < 0.05


def test_register_detects_a_reference_that_sees_the_back():
    v2, f2 = asymmetric_object()
    m = R.rasterize_camera(torch.tensor(v2), torch.tensor(f2), make_camera(cam_params(az=185.0)), 256).mask.numpy()
    reg = register_reference(R, v2, f2, m, prior_az=0.0)
    assert reg.flipped and reg.iou > 0.9
    assert reg.notes


def test_find_orientation_repairs_a_wrong_axis_convention():
    v2, f2 = asymmetric_object()
    m = R.rasterize_camera(torch.tensor(v2), torch.tensor(f2), make_camera(cam_params()), 256).mask.numpy()
    wrong = v2 @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float).T          # a y-up style frame
    M, iou = find_orientation(R, wrong, f2, m)
    assert iou > 0.9 and np.allclose(wrong @ M.T, v2, atol=1e-5)


def test_align_view_pulls_a_misplaced_side_view_back():
    cam_true = zero123pp_cameras()[1]
    fg = R.rasterize_camera(torch.tensor(V), torch.tensor(F), cam_true, 160).mask.numpy()
    off = cam_true.with_intrinsics(f=cam_true.f * 1.12, cx=0.06, cy=-0.05)
    before = (R.rasterize_camera(torch.tensor(V), torch.tensor(F), off, 160).mask.numpy() & fg).sum() / (
        (R.rasterize_camera(torch.tensor(V), torch.tensor(F), off, 160).mask.numpy() | fg).sum())
    al = align_view(R, V, F, fg, off)
    assert al.iou > before + 0.05 and al.iou > 0.93


def test_estimate_foreground_on_flat_background():
    img = np.full((64, 64, 3), 0.5, np.float32)
    img[16:48, 20:44] = (0.9, 0.2, 0.1)
    img[28:36, 28:36] = 0.5                               # a hole that matches the background colour
    fg = estimate_foreground(img)
    assert fg[32, 24] and not fg[2, 2] and fg[32, 32]     # the enclosed hole counts as foreground


def _views(blur=2):
    ref_cam = make_camera(cam_params(az=6.0, el=12.0, d=7.0, fov=20.0, cx=0.0, cy=0.0))
    ref_img, ref_m = render_gt(R, V, F, ref_cam, 768)
    views = [ViewSource(ref_img, ref_cam, 8.0, alpha=ref_m.astype(np.float32), name="artwork")]
    for i, c in enumerate(zero123pp_cameras(azimuth0=6.0)):
        img, _ = render_gt(R, V, F, c, 320, blur=blur)
        views.append(ViewSource(img, c, 1.0, name=f"side{i}", harmonize=True))
    return views


def _bake(views, size=512):
    nv, nf, uv, vm, _ = unwrap(V, F, size, "builtin")
    normals = vertex_normals(V, F)[vm]
    res = bake_texture(R, nv, nf, uv, normals, views, size=size)
    return nv, nf, uv, res


def _novel_psnr(nv, nf, uv, tex, angles):
    out = []
    for az, el in angles:
        cam = orbit_camera(az, el, 5.0, 24.0)
        gt, m = render_gt(R, V, F, cam, 256)
        img, m2 = render_textured(R, nv, nf, uv, tex, cam, 256, bg=(0.5, 0.5, 0.5))
        out.append(psnr(img, gt, m & m2))
    return float(np.mean(out))


def test_bake_reproduces_the_object_from_novel_views():
    nv, nf, uv, res = _bake(_views())
    assert res.texture.shape == (512, 512, 3) and res.texture.dtype == np.uint8
    p = _novel_psnr(nv, nf, uv, res.texture, [(50, 15), (200, 0), (-60, 25), (125, -20)])
    assert p > 22.0
    assert res.usage["artwork"] > 0.1                       # the artwork really is the main source at the front


def test_artwork_dominates_where_it_sees_and_stays_sharp():
    nv, nf, uv, res = _bake(_views())
    p_front = _novel_psnr(nv, nf, uv, res.texture, [(6, 12)])
    assert p_front > 24.0


def test_without_side_views_every_texel_is_still_filled():
    nv, nf, uv, res = _bake(_views()[:1])
    tex = res.texture.astype(int)
    covered = res.occupied
    assert (tex[covered].sum(-1) == 0).mean() < 0.01        # no black holes inside the charts
    assert res.seen[covered].mean() < 1.0                   # ...some of it is filled in, honestly reported


def test_gutters_are_padded_so_bilinear_filtering_never_bleeds_black():
    _, _, _, res = _bake(_views())
    occ = res.occupied
    from scipy.ndimage import binary_dilation

    ring = binary_dilation(occ, iterations=3) & ~occ
    assert (res.texture[ring].sum(-1) == 0).mean() < 0.02


def test_harmonisation_fixes_a_colour_cast_in_generated_views():
    views = _views()
    for v in views[1:]:
        v.image = np.clip(v.image * np.array([1.15, 0.9, 0.85]) + 0.03, 0, 1)
    nv, nf, uv, res_h = _bake(views)
    for v in views[1:]:
        v.harmonize = False
    _, _, _, res_n = _bake(views)
    ang = [(200, 0), (-60, 25), (125, -20)]
    assert _novel_psnr(nv, nf, uv, res_h.texture, ang) > _novel_psnr(nv, nf, uv, res_n.texture, ang) + 0.5
