import numpy as np
import pytest
from PIL import Image, ImageDraw

from aura_white.image import prepare_image
from aura_white.mesh import signed_volume, surface_nets
from aura_white.volume import build_volume, grid_to_world

THR = 25.0
R = 0.87


def scene(p):
    """sphere + torus + thin rod, as a steep exp-density field like the real decoder produces."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    sph = 0.35 - np.sqrt((x + 0.3) ** 2 + y ** 2 + z ** 2)
    tor = 0.10 - np.sqrt((np.sqrt((x - 0.25) ** 2 + z ** 2) - 0.30) ** 2 + (y + 0.15) ** 2)
    rod = 0.035 - np.sqrt((x - 0.1) ** 2 + (z + 0.05) ** 2)
    rod = np.where(np.abs(y) < 0.6, rod, -1)
    sdf = np.maximum(np.maximum(sph, tor), rod)
    return (THR * np.exp(np.clip(40.0 * sdf, -30, 30))).astype(np.float32)


def mesh_of(vol):
    v, f = surface_nets(np.pad(vol, 1, constant_values=0.0), THR)
    return grid_to_world(v, vol.shape[0], R, pad=1), f


def test_adaptive_matches_dense_and_saves_work():
    dense, sd = build_volume(scene, R, 128, THR, adaptive=False, chunk=200000)
    fast, sa = build_volume(scene, R, 128, THR, adaptive=True, chunk=200000)
    assert sd["mode"] == "dense" and sa["mode"] == "adaptive"
    assert sa["fraction"] < 0.4
    vd, fd = mesh_of(dense)
    va, fa = mesh_of(fast)
    assert len(vd) == len(va) and len(fd) == len(fa)
    assert np.abs(vd - va).max() < 1e-4
    assert signed_volume(va, fa) > 0


def test_adaptive_efficiency_at_full_resolution_shape():
    _, s = build_volume(scene, R, 256, THR, adaptive=True, chunk=400000)
    assert s["fraction"] < 0.15, s


def test_dense_ordering_is_xyz():
    # density increases with +x only: volume[i,:,:] must grow along axis 0
    vol, _ = build_volume(lambda p: p[:, 0].copy(), 1.0, 16, 0.0, adaptive=False, chunk=1000)
    assert np.all(np.diff(vol[:, 3, 5]) > 0) and np.allclose(vol[:, 0, 0], vol[:, 7, 9])
    vol2, _ = build_volume(lambda p: p[:, 2].copy(), 1.0, 16, 0.0, adaptive=False, chunk=1000)
    assert np.all(np.diff(vol2[3, 5, :]) > 0)


def studio_photo(bg=(235, 235, 235), size=(320, 240)):
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.ellipse((90, 50, 230, 190), fill=(200, 40, 40))
    d.rectangle((140, 120, 180, 200), fill=(30, 60, 200))
    return im


def test_border_matting_isolates_subject_and_frames_it():
    p = prepare_image(studio_photo(), remove_bg=True, foreground_ratio=0.8, matting="border")
    assert p.matting == "border" and not p.warnings
    assert p.rgb.shape[0] == p.rgb.shape[1] and p.rgb.dtype == np.float32
    a = np.asarray(p.cutout)[..., 3]
    assert a[0, 0] == 0 and a.max() == 255
    ys, xs = np.nonzero(a > 128)
    frac = max(ys.max() - ys.min(), xs.max() - xs.min()) / p.rgb.shape[0]
    assert 0.74 < frac < 0.86
    assert np.allclose(p.rgb[0, 0], 0.5)  # background composited to mid-grey


def test_existing_alpha_is_respected():
    im = studio_photo().convert("RGBA")
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).ellipse((90, 50, 230, 190), fill=255)
    im.putalpha(m)
    p = prepare_image(im, remove_bg=True)
    assert p.matting == "alpha"


def test_no_removal_pads_to_square():
    p = prepare_image(studio_photo(), remove_bg=False)
    assert p.matting == "none" and p.rgb.shape[0] == p.rgb.shape[1] == 320


def test_busy_background_warns_instead_of_crashing():
    rng = np.random.RandomState(0)
    noise = Image.fromarray(rng.randint(0, 255, (200, 200, 3)).astype(np.uint8))
    p = prepare_image(noise, remove_bg=True, matting="border")
    assert p.warnings and p.rgb.shape[2] == 3


@pytest.mark.parametrize("src", ["bytes", "array", "path"])
def test_input_types(tmp_path, src):
    im = studio_photo()
    if src == "bytes":
        import io
        b = io.BytesIO()
        im.save(b, "PNG")
        x = b.getvalue()
    elif src == "array":
        x = np.asarray(im)
    else:
        x = tmp_path / "a.png"
        im.save(x)
    assert prepare_image(x, matting="border").rgb.shape[2] == 3
