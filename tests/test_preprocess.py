import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from aura_white.image import load_image, prepare_image


def scene(bg=(240, 240, 240), size=(400, 300)):
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.ellipse((120, 70, 280, 230), fill=(200, 30, 40))
    d.rectangle((160, 120, 240, 170), fill=(20, 60, 200))
    return im


def test_plain_background_is_removed_and_framed():
    p = prepare_image(scene(), True, 0.85)
    assert p.matting in ("exact", "border", "rembg")
    assert p.rgb.shape[0] == p.rgb.shape[1]
    a = np.asarray(p.cutout.getchannel("A")) > 128
    ys, xs = np.where(a)
    side = p.cutout.size[0]
    assert abs((xs.max() - xs.min() + 1) / side - 0.85) < 0.06 or abs((ys.max() - ys.min() + 1) / side - 0.85) < 0.06
    assert np.allclose(p.rgb[2, 2], 0.5), "background becomes neutral gray"
    centre = p.rgb[side // 2, side // 2]
    assert centre[2] > 0.6 and centre[0] < 0.3, "object colours preserved"


def test_dark_and_coloured_backgrounds():
    for bg in [(15, 15, 20), (30, 140, 60), (250, 250, 250)]:
        p = prepare_image(scene(bg), True)
        a = np.asarray(p.cutout.getchannel("A")) > 128
        assert 0.15 < a.mean() < 0.8, bg


def test_alpha_is_respected_and_untouched():
    im = scene().convert("RGBA")
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).ellipse((120, 70, 280, 230), fill=255)
    im.putalpha(m)
    p = prepare_image(im, True)
    assert p.matting == "alpha"


def test_busy_background_warns_but_does_not_crash():
    rng = np.random.RandomState(0)
    noise = Image.fromarray(rng.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    d = ImageDraw.Draw(noise)
    d.ellipse((120, 70, 280, 230), fill=(255, 255, 255))
    p = prepare_image(noise, True)
    assert p.rgb.shape[0] == p.rgb.shape[1]
    assert p.matting in ("border", "none", "rembg")


@pytest.mark.parametrize("mode", ["L", "LA", "P", "CMYK", "I;16", "RGBA", "1"])
def test_odd_modes_load(mode):
    base = scene().convert("L" if mode in ("L", "1", "I;16") else "RGB")
    try:
        im = base.convert(mode)
    except Exception:
        pytest.skip("PIL cannot make this mode here")
    out = load_image(im)
    assert out.mode in ("RGB", "RGBA")
    prepare_image(im, True)


def test_bytes_numpy_and_no_bg_removal():
    buf = io.BytesIO()
    scene().save(buf, "PNG")
    assert load_image(buf.getvalue()).size == (400, 300)
    assert load_image(np.zeros((10, 20, 3), np.float32)).size == (20, 10)
    p = prepare_image(scene(), remove_bg=False)
    assert p.matting == "none" and p.rgb.shape[:2] == (400, 400)


def test_exif_orientation_applied():
    im = scene()
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    assert load_image(buf.getvalue()).size == (300, 400)


def test_huge_image_is_downscaled():
    big = Image.new("RGB", (5000, 3000), (240, 240, 240))
    assert max(load_image(big).size) <= 2048
