
import numpy as np
import pytest
import torch
import trimesh

from aura_white.raster import get_raster, interpolate, orbit_camera, transform_camera, normalize_mesh, zero123pp_cameras
from aura_white.raster.torch_raster import rasterize_screen


def test_single_triangle_covers_its_area():
    xy = torch.tensor([[2.0, 2.0], [14.0, 3.0], [6.0, 13.0]])
    tri, bary, depth = rasterize_screen(xy, torch.ones(3), torch.tensor([[0, 1, 2]]), 16, 16)
    hit = tri >= 0
    assert abs(int(hit.sum()) - 64) <= 2
    assert torch.allclose(bary[hit].sum(-1), torch.ones(int(hit.sum())), atol=1e-5)
    assert torch.isinf(depth[~hit]).all()


def test_zbuffer_matches_brute_force():
    g = torch.Generator().manual_seed(3)
    V = torch.rand(40, 2, generator=g) * 28 + 2
    W = torch.rand(40, generator=g) * 4 + 1
    T = torch.randint(0, 40, (25, 3), generator=g)
    tri, _, depth = rasterize_screen(V, W, T, 32, 32)
    Vn, Wn = V.double().numpy(), W.double().numpy()
    best_t = -np.ones((32, 32), int)
    best_d = np.full((32, 32), np.inf)
    for f, (a, b, c) in enumerate(T.numpy()):
        p0, p1, p2 = Vn[a], Vn[b], Vn[c]
        ar = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(ar) < 1e-12:
            continue
        for y in range(32):
            for x in range(32):
                cx, cy = x + .5, y + .5
                b0 = ((p1[0] - cx) * (p2[1] - cy) - (p1[1] - cy) * (p2[0] - cx)) / ar
                b1 = ((cx - p0[0]) * (p2[1] - p0[1]) - (cy - p0[1]) * (p2[0] - p0[0])) / ar
                b2 = 1 - b0 - b1
                if min(b0, b1, b2) >= -1e-6:
                    d = 1 / (b0 / Wn[a] + b1 / Wn[b] + b2 / Wn[c])
                    if d < best_d[y, x] - 1e-9 or (abs(d - best_d[y, x]) <= 1e-9 and f < best_t[y, x]):
                        best_d[y, x], best_t[y, x] = d, f
    both = (tri.numpy() >= 0) & (best_t >= 0)
    assert ((tri.numpy() >= 0) == (best_t >= 0)).mean() > 0.995
    assert (tri.numpy()[both] == best_t[both]).mean() > 0.995
    assert np.abs(depth.numpy()[both] - best_d[both]).max() < 1e-3


def test_sphere_silhouette_and_depth():
    sph = trimesh.creation.icosphere(subdivisions=4)
    v, f = torch.tensor(sph.vertices, dtype=torch.float32), torch.tensor(sph.faces)
    R = get_raster("cpu")
    cam = orbit_camera(20, 15, 4.0, 30.0)
    out = R.rasterize_camera(v, f, cam, 160)
    ys, xs = np.meshgrid(np.arange(160) + .5, np.arange(160) + .5, indexing="ij")
    nx, ny = xs / 160 * 2 - 1, 1 - ys / 160 * 2
    d = np.stack([nx / cam.f, ny / cam.f, -np.ones_like(nx)], -1) @ cam.c2w[:3, :3].T
    o = cam.c2w[:3, 3]
    a, b, c = (d ** 2).sum(-1), 2 * (d * o).sum(-1), (o ** 2).sum() - 1
    disc = b * b - 4 * a * c
    hit = disc > 0
    m = out.mask.numpy()
    assert (m & hit).sum() / (m | hit).sum() > 0.99
    t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    both = m & hit
    assert np.abs(out.depth.numpy()[both] - t[both]).mean() < 2e-3


def test_perspective_correct_interpolation():
    # a quad on the plane x=0 seen at an angle: interpolating world position must be exact per pixel
    v = torch.tensor([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], dtype=torch.float32)
    f = torch.tensor([[0, 1, 2], [0, 2, 3]])
    R = get_raster("cpu")
    cam = orbit_camera(50, 10, 3.0, 40.0)
    out = R.rasterize_camera(v, f, cam, 64)
    P = interpolate(v, f, out)
    m = out.mask
    assert int(m.sum()) > 100
    assert P[m][:, 0].abs().max() < 1e-4                                  # stays on the plane
    # re-project the interpolated point: must land on its own pixel centre
    pc = torch.tensor(cam.w2c, dtype=torch.float32)
    q = P[m] @ pc[:3, :3].T + pc[:3, 3]
    px = (cam.f * q[:, 0] / -q[:, 2] * .5 + .5) * 64
    py = (.5 - cam.f * q[:, 1] / -q[:, 2] * .5) * 64
    ii, jj = torch.nonzero(m, as_tuple=True)
    assert (px - (jj + .5)).abs().max() < 0.05 and (py - (ii + .5)).abs().max() < 0.05


def test_uv_raster_fills_unit_square():
    uv = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
    tris = torch.tensor([[0, 1, 2], [0, 2, 3]])
    out = get_raster("cpu").rasterize_uv(uv, tris, 32)
    assert bool(out.mask.all())


def test_camera_helpers():
    cams = zero123pp_cameras()
    assert len(cams) == 6
    assert np.allclose(np.linalg.norm(cams[0].position), 4.0)
    assert cams[0].position[2] > 0 > cams[1].position[2]                # elevations +20 / -10
    v, c, s = normalize_mesh(np.random.rand(50, 3).astype(np.float32) * 5 + 2)
    assert np.linalg.norm(v, axis=1).max() == pytest.approx(1.0, abs=1e-5)
    cam = orbit_camera(30, 20, 4.0)
    cam2 = transform_camera(cam, np.array([0.5, -0.2, 0.1]), 2.0)
    p = np.array([[0.3, 0.1, -0.2]])
    a = (cam.w2c[:3, :3] @ p.T).T + cam.w2c[:3, 3]
    b = (cam2.w2c[:3, :3] @ ((p - [0.5, -0.2, 0.1]) * 2.0).T).T + cam2.w2c[:3, 3]
    assert np.allclose(a[:, :2] / -a[:, 2:], b[:, :2] / -b[:, 2:], atol=1e-9)   # same pixel


def test_get_raster_falls_back_to_torch_on_cpu():
    r = get_raster("cpu", "auto")
    assert r.backend == "torch"
    with pytest.raises(RuntimeError):
        get_raster("cpu", "nvdiffrast")
