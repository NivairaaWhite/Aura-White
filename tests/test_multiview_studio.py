import pytest
from synth import FakeGeometry, make_artwork, make_asymmetric_object, psnr, render_gt

from aura_white.raster import get_raster, orbit_camera
from aura_white.studio import AuraStudio, StudioOptions
from aura_white.texture.preview import render_textured

R = get_raster("cpu")
V, F = make_asymmetric_object(3)


def studio(**kw):
    return AuraStudio(geometry=FakeGeometry(V, F), options=StudioOptions(quality="draft", texture_size=512,
                                                                        target_faces=6000, normal_map=False,
                                                                        complete=False, repair=True, **kw),
                      device="cpu")


def test_multiview_needs_at_least_two_views():
    front = make_artwork(R, V, F, az=0.0, el=10.0, size=256)
    with pytest.raises(ValueError):
        studio().generate_multiview({"front": front})


def test_multiview_registers_every_named_view_well():
    views = {"front": make_artwork(R, V, F, az=0.0, el=10.0, size=512),
            "right": make_artwork(R, V, F, az=90.0, el=10.0, size=512),
            "back": make_artwork(R, V, F, az=180.0, el=10.0, size=512)}
    res = studio().generate_multiview(views)
    r = res.report
    assert r["mode"] == "multiview" and set(r["views"]) == set(views)
    for name, iou in r["view_iou"].items():
        assert iou > 0.9, f"{name} registered poorly (IoU {iou})"
    assert res.albedo.shape == (512, 512, 3)


def test_multiview_texture_beats_a_single_front_view_on_a_novel_side_angle():
    """The whole point of joint reconstruction: a side/back angle no single photo showed should look far more
    correct once that side's own photo was actually used, than when only the front photo was available."""
    front = make_artwork(R, V, F, az=0.0, el=10.0, size=512)
    back = make_artwork(R, V, F, az=180.0, el=10.0, size=512)

    single = studio().generate(front)
    multi = studio().generate_multiview({"front": front, "back": back})

    novel_cam = orbit_camera(160.0, 10.0, 9.0, 28.0)          # close to the back view, far from the front one
    gt, gm = render_gt(R, V, F, novel_cam, 256)

    img_single, m_single = render_textured(R, single.verts, single.faces, single.uv, single.albedo, novel_cam, 256,
                                           bg=(0.5, 0.5, 0.5))
    img_multi, m_multi = render_textured(R, multi.verts, multi.faces, multi.uv, multi.albedo, novel_cam, 256,
                                         bg=(0.5, 0.5, 0.5))

    psnr_single = psnr(img_single, gt, gm & m_single)
    psnr_multi = psnr(img_multi, gt, gm & m_multi)
    assert psnr_multi > psnr_single + 3.0, (
        f"multiview ({psnr_multi:.1f} dB) should look much closer to the truth on the back angle than "
        f"front-only ({psnr_single:.1f} dB)")


def test_multiview_azimuth_guess_recognises_common_view_names():
    st = studio()
    assert st._AZIMUTH_GUESS["front"] == 0.0
    assert st._AZIMUTH_GUESS["back"] == 180.0
    assert st._AZIMUTH_GUESS["right"] == 90.0
    assert st._AZIMUTH_GUESS["left"] == 270.0


def test_multiview_accepts_explicit_azimuths_for_unrecognised_names():
    views = {"cam_a": make_artwork(R, V, F, az=0.0, el=10.0, size=384),
            "cam_b": make_artwork(R, V, F, az=120.0, el=10.0, size=384)}
    res = studio().generate_multiview(views, azimuths={"cam_a": 0.0, "cam_b": 120.0}, primary="cam_a")
    assert res.report["view_iou"]["cam_b"] > 0.85
