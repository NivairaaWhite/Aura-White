import numpy as np
import pytest
from PIL import Image, ImageDraw

from aura_white.image import matte
from aura_white.image.preprocess import prepare_image

pytest.importorskip("scipy")


def scene_dark(bg=255):
    im = Image.new("RGB", (300, 300), (bg, bg, bg))
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 240, 240), fill=(30, 20, 40))
    d.ellipse((130, 130, 170, 170), fill=(bg, bg, bg))          # a real hole: dark all around
    d.line((150, 240, 150, 295), fill=(120, 120, 125), width=5)  # hairline chain hanging off the bottom
    return np.asarray(im)


def test_exact_matte_keeps_hairline_and_hole():
    a = scene_dark()
    r = matte.plain_background_matte(a)
    assert r is not None
    assert r.hard[150, 100]                                     # body
    assert not r.hard[150, 150], "enclosed background must stay a hole"
    assert r.holes >= 1
    assert r.hard[280, 150], "the 5 px wire must survive the matte"
    assert not r.hard[5, 5]
    assert r.alpha.dtype == np.float32 and 0.0 <= r.alpha.min() and r.alpha.max() <= 1.0


def test_white_on_white_object_is_not_swallowed():
    rng = np.random.default_rng(0)
    base = np.full((320, 320, 3), 254.0)
    yy, xx = np.mgrid[:320, :320]
    body = ((yy - 160) ** 2 + (xx - 160) ** 2) < 110 ** 2
    tex = rng.normal(0, 3.0, base.shape[:2])                    # pale fur: as bright as the background but not flat
    base[body] = np.clip(250 + tex[body, None], 0, 255)
    # a broken outline: a gap that would let a naive flood fill leak in
    ring = (((yy - 160) ** 2 + (xx - 160) ** 2) > 108 ** 2) & (((yy - 160) ** 2 + (xx - 160) ** 2) < 111 ** 2)
    ring &= ~((xx > 150) & (xx < 170) & (yy < 160))
    base[ring] = 200
    r = matte.plain_background_matte(np.clip(base, 0, 255).astype(np.uint8))
    assert r is not None
    assert r.hard[160, 160] and r.hard[160, 230]
    assert r.hard[60, 160], "pale interior reached through the gap in the outline must stay foreground"
    assert not r.hard[5, 5] and not r.hard[315, 315]


def test_declines_busy_backgrounds():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (200, 200, 3)).astype(np.uint8)
    assert matte.plain_background_matte(noise) is None


def test_prepare_image_uses_exact_matte_and_frames():
    p = prepare_image(Image.fromarray(scene_dark()))
    assert p.matting == "exact"
    a = np.asarray(p.cutout.getchannel("A"))
    assert (a > 128).mean() > 0.2
