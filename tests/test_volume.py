import numpy as np

from aura_white.mesh import signed_volume, surface_nets
from aura_white.volume import build_density_volume, upsample_trilinear

R_WORLD = 0.87


def blob_density(p):
    """Smooth analytic 'density': exp of a signed-distance-ish field for sphere + torus + a bar."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    sphere = 0.30 - np.sqrt((x + 0.35) ** 2 + y ** 2 + z ** 2)
    torus = 0.09 - np.sqrt((np.sqrt((x - 0.30) ** 2 + y ** 2) - 0.28) ** 2 + z ** 2)
    bar = 0.06 - np.maximum(np.abs(x - 0.0), np.maximum(np.abs(y + 0.55), np.abs(z))) * 1.0
    sdf = np.maximum(np.maximum(sphere, torus), bar)
    return np.exp(np.clip(sdf * 60.0, -30, 30) + np.log(25.0)).astype(np.float32)  # == 25 on the surface


def test_upsample_is_exact_for_linear_fields():
    n = 9
    t = np.linspace(0, 1, n, dtype=np.float32)
    x, y, z = np.meshgrid(t, t, t, indexing="ij")
    v = 2 * x + 3 * y - z
    big = upsample_trilinear(v, 33)
    t2 = np.linspace(0, 1, 33, dtype=np.float32)
    X, Y, Z = np.meshgrid(t2, t2, t2, indexing="ij")
    assert np.abs(big - (2 * X + 3 * Y - Z)).max() < 1e-5


def test_adaptive_matches_dense_and_saves_queries():
    fn = lambda p: blob_density(p)
    dense, sd = build_density_volume(fn, 128, R_WORLD, 25.0, adaptive=False)
    adapt, sa = build_density_volume(fn, 128, R_WORLD, 25.0, adaptive=True)
    assert sa["adaptive"] and sa["queries"] < 0.35 * sd["queries"]
    vd, fd = surface_nets(dense, 25.0)
    va, fa = surface_nets(adapt, 25.0)
    assert len(fd) > 3000
    assert abs(len(fa) - len(fd)) / len(fd) < 0.02, "same surface, same triangle count"
    # inside/outside classification must agree everywhere except (at most) a hair
    disagree = np.mean((dense > 25.0) != (adapt > 25.0))
    assert disagree < 1e-4
    voldiff = abs(signed_volume(va, fa) - signed_volume(vd, fd)) / abs(signed_volume(vd, fd))
    assert voldiff < 0.01
