import numpy as np
import pytest
from PIL import Image, ImageDraw

import prims
from aura_white.gen3d import controller as C
from aura_white.gen3d import fusion as F
from aura_white.gen3d import intake as I
from aura_white.gen3d import meshutil as M
from aura_white.gen3d import outline as O
from aura_white.gen3d import shapegen as S
from aura_white.gen3d import views as V
from aura_white.gen3d.hardware import Hardware
from aura_white.gen3d.recipe import Recipe

TRUTH = prims.figure()
SPEC = V.ViewSpec.parse


def masks(names, truth=TRUTH):
    return [(SPEC(n), V.render_maps(*truth, V.ViewCam(SPEC(n).az, SPEC(n).el, 192, 80.0, 0.0, 0.2))[0]) for n in names]


def imperfect():
    """The truth without its tail, and with a short arm: what a generator typically hands back."""
    return prims.merge(prims.uv_sphere(24, 48, (0.45, 0.8, 0.30)), prims.translate(prims.box((0.45, 0.12, 0.12), 6), (0.45, 0.25, 0.0)),
                       prims.translate(prims.uv_sphere(14, 28, 0.22), (0.0, 1.0, 0.0)),
                       prims.translate(prims.uv_sphere(10, 20, 0.07), (0.0, 1.0, 0.24)))


# ------------------------------------------------------------------------------------------ fusion
def test_fusion_leaves_a_mesh_that_already_agrees_untouched():
    cons = F.build_constraints(*TRUTH, masks(["front", "left"]))
    r = F.fuse(*TRUTH, cons, grid=64)
    assert not r.applied and "untouched" in " ".join(r.notes)


def test_fusion_with_four_pictures_improves_agreement_without_wrecking_the_volume():
    gen = imperfect()
    cons = F.build_constraints(*gen, masks(["front", "left", "back", "right"]))
    r = F.fuse(*gen, cons, grid=96)
    assert r.applied and r.stats["iou_after"] > r.stats["iou_before"] + 0.005
    lo, side = M.grid_for(TRUTH[0])
    vt, vg, vf = (M.voxelize(*m, 64, lo, side) for m in (TRUTH, gen, (r.verts, r.faces)))
    iou = lambda a, b: (a & b).sum() / (a | b).sum()
    assert iou(vt, vf) > iou(vt, vg) - 0.05          # silhouette-consistent growth must not invent much volume


def test_fusion_refuses_pictures_that_cannot_belong_to_this_mesh():
    # a sphere "photographed" as a thin vertical bar: carving would remove nearly everything -> keep the mesh
    sphere = prims.uv_sphere(24, 48, 1.0)
    bar = np.zeros((192, 192), bool)
    bar[20:172, 90:102] = True
    cons = F.build_constraints(*sphere, [(SPEC("front"), bar), (SPEC("left"), bar)])
    r = F.fuse(*sphere, cons, grid=64)
    assert not r.applied and "registered" in " ".join(r.notes)


# ------------------------------------------------------------------------------------------ controller
def test_loop_rejects_slabs_and_repairs_a_rotated_frame():
    rec = Recipe.preset("standard").with_(candidates=2, max_rounds=3, fusion_grid=64)
    slab = prims.slab(1.4, 2.0, 0.05)
    Rz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    guidance = []

    def gen(call):
        guidance.append(call.guidance)
        if call.round == 0:
            return slab
        return TRUTH[0] @ Rz.T, TRUTH[1]

    res = C.run_quality_loop(gen, masks(["front"]), rec)
    assert res.passed and res.best.audit.flat_risk == 0 and res.best.audit.depth_ratio > 0.4
    assert res.best.rotation != 0                                   # the z-up output was turned upright
    assert max(guidance) > rec.shape_guidance                        # a flat round raises guidance for the next one
    assert any(h["profile"] != "solid" for h in res.history)         # the slabs were seen and recorded


def test_loop_reports_honestly_when_it_cannot_do_better():
    rec = Recipe.preset("draft").with_(candidates=1, max_rounds=2, fusion_grid=64)
    res = C.run_quality_loop(lambda call: prims.slab(1.4, 2.0, 0.05), masks(["front"]), rec)
    assert not res.passed and "flat" in res.reason


def test_flat_objects_pass_when_flat_is_expected():
    rec = Recipe.preset("draft").with_(candidates=1, max_rounds=1, fusion_grid=64, expect_solid=False, view_iou_target=0.5)
    res = C.run_quality_loop(lambda call: prims.slab(1.1, 1.9, 0.05), masks(["front"]), rec)
    assert res.best.audit.flat_risk == 0


def test_swapped_left_right_labels_are_detected_and_correct_ones_are_not():
    R = lambda az: V.render_maps(*TRUTH, V.ViewCam(az, 0.0, 192, 80.0, 0.0, 0.2))[0]
    correct = [(SPEC("front"), R(0)), (SPEC("left"), R(90)), (SPEC("right"), R(-90))]
    wrong = [(SPEC("front"), R(0)), (SPEC("left"), R(-90)), (SPEC("right"), R(90))]
    assert C.side_swap_gain(*TRUTH, correct) <= 0.0
    assert C.side_swap_gain(*TRUTH, wrong) > 0.05


def test_symmetric_silhouette_does_not_flip_the_object():
    v, f = prims.uv_sphere(20, 40, 1.0)
    view = [(SPEC("front"), V.render_maps(v, f, V.ViewCam(0, 0, 192, 80.0, 0, 0))[0])]
    _, rot, _ = C.align_frame(v, f, view)
    assert np.allclose(V.proper_rotations()[rot], np.eye(3))


# ------------------------------------------------------------------------------------------ outlines
def _figure_png(bg=(255, 255, 255), line=False):
    im = Image.new("RGB", (480, 640), bg)
    d = ImageDraw.Draw(im)
    c = lambda col: None if line else col
    d.ellipse((190, 60, 290, 170), fill=c((235, 190, 160)), outline=(20, 20, 20), width=4)
    d.polygon([(170, 180), (310, 180), (340, 420), (140, 420)], fill=c((200, 60, 70)), outline=(20, 20, 20))
    d.rectangle((170, 420, 230, 590), fill=c((50, 60, 140)), outline=(20, 20, 20), width=4)
    d.rectangle((250, 420, 310, 590), fill=c((50, 60, 140)), outline=(20, 20, 20), width=4)
    return im


def _truth_mask():
    m = Image.new("L", (480, 640), 0)
    d = ImageDraw.Draw(m)
    d.ellipse((190, 60, 290, 170), fill=255)
    d.polygon([(170, 180), (310, 180), (340, 420), (140, 420)], fill=255)
    d.rectangle((170, 420, 230, 590), fill=255)
    d.rectangle((250, 420, 310, 590), fill=255)
    return np.asarray(m) > 127


iou = lambda a, b: (a & b).sum() / (a | b).sum()


def test_outline_of_art_on_a_plain_background():
    o = O.extract_outline(_figure_png(), use_models=False)
    assert iou(o.hard, _truth_mask()) > 0.99 and o.score > 0.8


def test_outline_of_line_art():
    o = O.extract_outline(_figure_png(line=True), use_models=False)
    assert o.kind == "lineart" and iou(o.hard, _truth_mask()) > 0.97


def test_a_real_alpha_channel_always_wins_even_when_the_transparent_area_is_black():
    # regression: a saved cut-out has black under its transparency; background-based methods used to agree on a wrong rectangle
    t = _truth_mask()
    rgba = np.zeros((640, 480, 4), np.uint8)
    rgba[..., :3][t] = (200, 90, 60)
    rgba[..., 3][t] = 255
    rgba[100:560, 60:420, :3] = np.where(t[100:560, 60:420, None], rgba[100:560, 60:420, :3], 236)    # grey box inside the crop
    o = O.extract_outline(Image.fromarray(rgba, "RGBA"), use_models=False)
    assert o.method == "alpha" and iou(o.hard, t) > 0.99


def test_picture_without_a_subject_is_refused_not_turned_into_a_slab():
    noise = Image.fromarray(np.random.default_rng(1).integers(0, 255, (400, 400, 3), dtype=np.uint8))
    with pytest.raises(O.OutlineError):
        O.extract_outline(noise, use_models=False)
    o = O.extract_outline(noise, use_models=False, strict=False)
    assert o.method == "none"                                        # only when the caller explicitly accepts it


def test_busy_background_is_cut_out_when_opencv_is_available():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:640, 0:480]
    bg = np.stack([120 + 80 * np.sin(xx / 37.0) + rng.normal(0, 18, (640, 480)), 150 + 70 * np.cos(yy / 29.0) + rng.normal(0, 18, (640, 480)),
                   100 + 90 * np.sin((xx + yy) / 53.0) + rng.normal(0, 18, (640, 480))], -1).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(bg)
    im.paste(_figure_png(), (0, 0), Image.fromarray((_truth_mask() * 255).astype(np.uint8)))
    o = O.extract_outline(im, use_models=False)
    assert iou(o.hard, _truth_mask()) > 0.95


def test_border_flood_is_not_trusted_on_a_cluttered_border():
    rng = np.random.default_rng(0)
    clutter = Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8))
    _, feats = O.classify_image(np.asarray(clutter))
    assert feats["border_uniform"] < O.BORDER_FLOOD_MIN_UNIFORM


# ------------------------------------------------------------------------------------------ intake
def test_inputs_are_split_into_pictures_and_words(tmp_path):
    p = tmp_path / "knight_back.png"
    _figure_png().save(p)
    files, words = I.split_inputs([str(p), "a", "bronze knight"])
    assert len(files) == 1 and words == ["a", "bronze knight"]


def test_one_picture_becomes_the_front_view(tmp_path):
    p = tmp_path / "dragon.png"
    _figure_png().save(p)
    req = I.load_request([str(p)])
    assert req.mode == "image" and req.reference().spec.label == "front"


def test_several_pictures_are_labelled_by_name_and_front_comes_first(tmp_path):
    for n in ("hero_back.png", "hero_left.png", "hero_front.png"):
        _figure_png().save(tmp_path / n)
    req = I.load_request([str(tmp_path / n) for n in ("hero_back.png", "hero_left.png", "hero_front.png")])
    assert req.mode == "multi" and [v.spec.label for v in req.views] == ["front", "left", "back"]


def test_unlabelled_pictures_are_not_guessed(tmp_path):
    for n in ("a.png", "b.png"):
        _figure_png().save(tmp_path / n)
    with pytest.raises(ValueError) as e:
        I.load_request([str(tmp_path / "a.png"), str(tmp_path / "b.png")])
    assert "which side" in str(e.value)


def test_two_pictures_of_the_same_view_are_refused(tmp_path):
    for n in ("one.png", "two.png"):
        _figure_png().save(tmp_path / n)
    with pytest.raises(ValueError):
        I.load_request([f"back={tmp_path / 'one.png'}", f"back={tmp_path / 'two.png'}"])


def test_text_only_request_has_no_views():
    req = I.load_request(["a bronze dragon statue"])
    assert req.mode == "text" and req.text == "a bronze dragon statue" and req.reference() is None


def test_nothing_to_build_from_is_an_error():
    with pytest.raises(ValueError):
        I.load_request([""])


# ------------------------------------------------------------------------------------------ shape back-end choice
def _views():
    img = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
    m = np.ones((64, 64), bool)
    return [I.ViewInput(SPEC(n), img, m, None) for n in ("front", "left", "back", "right")]


HW = Hardware("cuda", "8 GB card", 8, 7.6, 32, 24, True, True)


def test_without_a_neural_generator_it_refuses_instead_of_making_a_slab():
    with pytest.raises(S.NoBackend) as e:
        S.choose_backend(_views(), HW)
    msg = str(e.value)
    assert "never produced silently" in msg and "setup --hunyuan" in msg


def test_basic_fallbacks_are_opt_in_and_labelled():
    ch = S.choose_backend(_views(), HW, allow_basic=True)
    assert ch.backend.tier in ("hull", "basic") and any("fall-back" in n for n in ch.notes)


def test_a_visual_hull_needs_depth_information():
    two = [v for v in _views() if v.spec.label in ("front", "back")]
    assert not S.HullShape().probe(HW, two).ok
    assert S.HullShape().probe(HW, _views()).ok


def test_the_hull_fallback_builds_a_real_solid():
    from aura_white.gen3d.audit import audit_mesh
    views = [I.ViewInput(SPEC(n), Image.new("RGBA", (192, 192)), m, None) for n, m in
             zip(("front", "left", "back", "right"), [mk for _, mk in masks(["front", "left", "back", "right"])])]
    v, f = S.HullShape().generate(S.GenCall(0, 384, 50, 5.0), views, None, lambda *_: None)
    a = audit_mesh(v, f)
    assert a.watertight and a.depth_ratio > 0.3


def test_call_supported_drops_arguments_a_vendor_version_does_not_have():
    def vendor(image, steps=10):
        return image, steps
    assert S.call_supported(vendor, image=1, steps=3, enable_pbar=False, octree=9) == (1, 3)
