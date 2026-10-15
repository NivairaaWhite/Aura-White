import json

import numpy as np
import pytest
from PIL import Image
from synth import FakeGeometry, FakeMultiView, make_artwork, make_object, psnr, render_gt

from aura_white.backends.base import MultiViewResult
from aura_white.raster import get_raster, orbit_camera
from aura_white.raster.camera import ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS
from aura_white.studio import AuraStudio, StudioOptions, PRESETS
from aura_white.texture.preview import render_textured

R = get_raster("cpu")
V, F = make_object(3)
OPTS = dict(quality="draft", texture_size=512, target_faces=4000, normal_map=False, fidelity="standard")


def novel_psnr(res, angles=((50, 15), (200, 0), (-60, 25), (125, -20))):
    out = []
    for az, el in angles:
        cam = orbit_camera(az, el, 5.0, 24.0)
        gt, m = render_gt(R, res.verts, res.faces, cam, 256)        # colour field lives on the normalised frame
        img, m2 = render_textured(R, res.verts, res.faces, res.uv, res.albedo, cam, 256, bg=(0.5, 0.5, 0.5))
        out.append(psnr(img, gt, m & m2))
    return float(np.mean(out))


@pytest.fixture(scope="module")
def artwork():
    return make_artwork(R, V, F)


def studio(**kw):
    geometry = kw.pop("geometry", FakeGeometry(V, F))
    multiview = kw.pop("multiview", FakeMultiView(R, V, F))
    return AuraStudio(geometry=geometry, multiview=multiview, options=StudioOptions(**{**OPTS, **kw}), device="cpu")


def test_studio_end_to_end_with_side_views(artwork, tmp_path):
    logs = []
    mv = FakeMultiView(R, V, F)
    res = studio(multiview=mv).generate(artwork, progress=lambda s, f, m: logs.append(s))
    r = res.report
    assert r["reference_iou"] > 0.95 and r["side_views"] and len(r["side_view_iou"]) == 6
    assert mv.unloaded                                                 # GPU memory is released between stages
    assert novel_psnr(res) > 22.0
    assert res.albedo.shape == (512, 512, 3) and len(res.faces) <= 4500
    assert {"artwork", "geometry", "registration", "texture"} <= set(logs)
    files = res.save_all(tmp_path, "asset")
    assert files["glb"].stat().st_size > 1000 and files["obj"].exists() and files["albedo"].exists()
    assert json.loads(files["report"].read_text())["texture_size"] == 512
    import trimesh

    g = list(trimesh.load(files["glb"]).geometry.values())[0]
    assert g.visual.material.baseColorTexture is not None


def test_studio_without_side_views_still_produces_a_complete_texture(artwork):
    res = studio(multiview="off").generate(artwork)
    assert not res.report["side_views"]
    assert any("side-view" in n for n in res.report["notes"])
    occ = res.albedo.sum(-1) > 0
    assert occ.mean() > 0.3
    assert novel_psnr(res, angles=((6, 10),)) > 21.0                    # the front, from the artwork, is right


def test_geometry_in_the_wrong_axis_convention_is_corrected_from_generated_views():
    """InstantMesh-style: mesh + the six views it was reconstructed from, but the axes are permuted/mirrored."""
    from synth import chamfer, make_asymmetric_object

    v2, f2 = make_asymmetric_object(3)
    imgs = []
    for a, e in zip(ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS):
        img, _ = render_gt(R, v2, f2, orbit_camera(a, e, 4.0, 30.0), 320, bg=0.5)
        imgs.append(img.astype(np.float32))
    views = MultiViewResult(imgs, None, "gray")
    M = np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], float)             # some other lab's convention
    wrong = (v2 @ M.T).astype(np.float32)
    art = make_artwork(R, v2, f2, az=0.0, el=20.0)
    res = studio(geometry=FakeGeometry(wrong, f2, absolute=True, views=views), multiview="off").generate(art)
    assert any("axes corrected" in n for n in res.report["notes"])
    assert chamfer(res.verts, v2) < 0.02                                # the original mesh is restored
    assert novel_psnr(res) > 20.0


def test_unusable_side_views_are_ignored_not_fatal(artwork):
    class Junk(FakeMultiView):
        def generate(self, image, **kw):
            rng = np.random.default_rng(0)
            return MultiViewResult([rng.random((320, 320, 3)).astype(np.float32) for _ in range(6)], None, "gray")

    res = studio(multiview=Junk(R, V, F)).generate(artwork)
    assert res.report["reference_iou"] > 0.9
    assert set(res.report["view_usage"]) == {"artwork"} or res.report["view_usage"].get("side1", 0) == 0
    assert novel_psnr(res, angles=((6, 10),)) > 21.0


def test_failing_geometry_backend_falls_back_when_auto():
    class Bad:
        name = "bad"

        def generate(self, *a, **k):
            raise RuntimeError("out of memory")

    # explicit backends propagate errors...
    with pytest.raises(RuntimeError):
        studio(geometry=Bad(), multiview="off").generate(Image.new("RGB", (64, 64), "white"))


def test_presets_and_cpu_downgrade():
    o = StudioOptions(quality="ultra")
    assert o.resolved(True)["texture_size"] == 4096
    r = o.resolved(False)
    assert r["texture_size"] <= 2048 and r["target_faces"] <= 60_000
    assert StudioOptions(quality="draft", texture_size=777).resolved(True)["texture_size"] == 777
    with pytest.raises(ValueError):
        StudioOptions(quality="nope").resolved(True)
    assert set(PRESETS) == {"draft", "standard", "high", "ultra"}


def test_builtin_geometry_path_runs_without_any_heavy_model():
    from aura_white.backends import AuraGeometry
    from aura_white.pipeline import AuraWhite

    from PIL import ImageDraw

    im = Image.new("RGB", (256, 256), (235, 235, 235))
    ImageDraw.Draw(im).ellipse((60, 40, 200, 220), fill=(200, 60, 50))
    st = AuraStudio(geometry=AuraGeometry("cpu", None, 64, "auto", engine=AuraWhite.from_random()), multiview="off",
                    options=StudioOptions(quality="draft", texture_size=256, target_faces=3000, fidelity="standard"),
                    device="cpu")
    res = st.generate(im)
    assert res.albedo.shape == (256, 256, 3) and res.report["geometry"] == "aura"
    assert len(res.faces) <= 3500


def test_default_is_source_locked_and_reproduces_the_artwork(artwork):
    """The shipped default: the supplied picture is the authority for the reference view."""
    assert StudioOptions().fidelity == "source_locked"
    res = studio(multiview="off", fidelity="source_locked").generate(artwork)
    r = res.report
    sl = r["refine"]["source_locked"]
    assert sl["closed"] and sl["mask_coverage_iou"] > 0.999
    assert r["source_fidelity_iou"] > 0.995 and r["source_fidelity_psnr_db"] > 30.0
    assert res.albedo.shape[2] == 3 and len(res.uv) == len(res.verts)
