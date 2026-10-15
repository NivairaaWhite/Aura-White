import numpy as np
import pytest
from PIL import Image, ImageDraw

from aura_white import AuraWhite
from aura_white.pipeline import orient_vertices, save_gif
from aura_white.mesh import signed_volume

trimesh = pytest.importorskip("trimesh")


@pytest.fixture(scope="module")
def engine():
    return AuraWhite.from_random()


@pytest.fixture(scope="module")
def photo():
    im = Image.new("RGB", (200, 160), (235, 235, 235))
    d = ImageDraw.Draw(im)
    d.ellipse((50, 30, 150, 130), fill=(210, 40, 40))
    return im


def make(engine, photo, **kw):
    from aura_white.image import prepare_image
    prep = prepare_image(photo)
    code = engine.encode_image(prep.rgb)
    thr = engine.auto_threshold(code)
    return engine.generate(prep, threshold=thr, **kw)


def test_end_to_end_random_model_gives_valid_mesh(engine, photo):
    r = make(engine, photo, resolution=64)
    assert len(r.faces) > 100 and r.colors.shape == (len(r.vertices), 3) and r.colors.dtype == np.uint8
    assert np.abs(r.vertices).max() <= engine.cfg.radius * 1.05
    assert r.faces.max() < len(r.vertices) and r.faces.min() >= 0
    assert signed_volume(r.vertices, r.faces) > 0
    assert set(r.timings) >= {"preprocess", "encode", "density", "mesh", "colors", "total"}


def test_adaptive_and_dense_agree_on_the_model(engine, photo):
    a = make(engine, photo, resolution=112, adaptive=True)
    d = make(engine, photo, resolution=112, adaptive=False)
    assert a.stats["adaptive"] and not d.stats["adaptive"]
    # a random network has structure at every scale, so nearly the whole volume is 'active' here;
    # the real saving is asserted on smooth fields in test_volume.py. Worst case = dense + coarse pass.
    assert a.stats["queries"] <= d.stats["queries"] * 1.03
    assert abs(len(a.faces) - len(d.faces)) / max(len(d.faces), 1) < 0.05


def test_deterministic(engine, photo):
    a, b = make(engine, photo, resolution=48), make(engine, photo, resolution=48)
    assert np.array_equal(a.faces, b.faces) and np.allclose(a.vertices, b.vertices)


def test_low_memory_mode_same_result(engine, photo):
    a, b = make(engine, photo, resolution=48), make(engine, photo, resolution=48, low_memory=True)
    assert np.array_equal(a.faces, b.faces) and np.allclose(a.vertices, b.vertices, atol=1e-5)


def test_save_all_formats_and_orientation(engine, photo, tmp_path):
    r = make(engine, photo, resolution=48)
    files = r.save_all(tmp_path, "m", ("glb", "obj", "ply", "stl"))
    assert set(files) == {"glb", "obj", "ply", "stl"} and all(p.stat().st_size > 0 for p in files.values())
    m = trimesh.load(str(files["ply"]), process=True)
    assert m.is_watertight
    up_y = r.oriented("y")
    assert np.allclose(up_y[:, 1], r.vertices[:, 2]) and np.allclose(up_y[:, 2], r.vertices[:, 0])
    # rotation only: volume and handedness are preserved
    assert abs(signed_volume(up_y, r.faces) - signed_volume(r.vertices, r.faces)) < 1e-4
    z = orient_vertices(r.vertices, "z", 90.0)
    assert abs(signed_volume(z, r.faces) - signed_volume(r.vertices, r.faces)) < 1e-4


def test_turntable_preview(engine, photo, tmp_path):
    r = make(engine, photo, resolution=40)
    frames = engine.render_turntable(r, n_views=3, size=24)
    assert len(frames) == 3 and frames[0].shape == (24, 24, 3) and frames[0].dtype == np.uint8
    # a random model can render identical frames and Pillow merges duplicates, so use distinct ones
    distinct = [np.full((24, 24, 3), 40 * i, np.uint8) for i in range(3)]
    p = save_gif(distinct, tmp_path / "t.gif")
    assert Image.open(p).n_frames == 3


def test_no_surface_message(engine, photo):
    with pytest.raises(RuntimeError, match="No surface"):
        engine.generate(photo, threshold=1e30, resolution=32)


def test_floaters_removed_by_default(engine, photo):
    r = make(engine, photo, resolution=64, min_part_ratio=0.02)
    from aura_white.mesh import connected_components
    lab = connected_components(r.faces, len(r.vertices))
    assert len(np.unique(lab[np.unique(r.faces)])) >= 1
