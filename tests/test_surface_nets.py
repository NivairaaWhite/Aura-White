import numpy as np
import pytest

from aura_white.mesh import (connected_components, keep_large_components, signed_volume,
                             surface_nets, taubin_smooth, vertex_normals)


def grid(n, lo=-1.0, hi=1.0):
    t = np.linspace(lo, hi, n, dtype=np.float32)
    return np.meshgrid(t, t, t, indexing="ij")


def edge_stats(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    und = np.sort(e, axis=1)
    _, counts = np.unique(und, axis=0, return_counts=True)
    dir_set = {tuple(x) for x in e.tolist()}
    reverse_ok = all((b, a) in dir_set for a, b in e.tolist())
    return counts, reverse_ok


def euler(verts, faces):
    e = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    used = np.unique(faces)
    return len(used) - len(e) + len(faces)


def test_sphere_is_closed_oriented_and_accurate():
    x, y, z = grid(64)
    vol = 0.6 - np.sqrt(x * x + y * y + z * z)
    v, f = surface_nets(vol, 0.0)
    assert v.shape[0] > 1000 and f.shape[0] > 2000
    counts, reverse_ok = edge_stats(f)
    assert (counts == 2).all(), "every edge must be shared by exactly two faces (watertight)"
    assert reverse_ok, "orientation must be consistent"
    assert euler(v, f) == 2
    step = 2.0 / 63
    world = v * step - 1.0
    vol_est = signed_volume(world, f)
    assert vol_est > 0, "faces must point outward"
    assert abs(vol_est - 4 / 3 * np.pi * 0.6 ** 3) / (4 / 3 * np.pi * 0.6 ** 3) < 0.03
    r = np.linalg.norm(world, axis=1)
    assert abs(r.mean() - 0.6) < 0.005 and r.std() < 0.01


def test_normals_point_outward():
    x, y, z = grid(48)
    vol = 0.5 - np.sqrt(x * x + y * y + z * z)
    v, f = surface_nets(vol, 0.0)
    world = v * (2.0 / 47) - 1.0
    n = vertex_normals(world, f)
    assert np.mean(np.einsum("ij,ij->i", n, world / np.linalg.norm(world, axis=1, keepdims=True))) > 0.98


def test_torus_genus_one():
    x, y, z = grid(72)
    vol = 0.25 - np.sqrt((np.sqrt(x * x + y * y) - 0.55) ** 2 + z * z)
    v, f = surface_nets(vol, 0.0)
    counts, _ = edge_stats(f)
    assert (counts == 2).all()
    assert euler(v, f) == 0
    assert signed_volume(v * (2 / 71) - 1, f) > 0


def test_inverted_field_gives_no_surface_or_flipped_volume():
    v, f = surface_nets(np.full((8, 8, 8), -1.0, np.float32), 0.0)
    assert f.shape[0] == 0


def test_any_level_and_anisotropic_shape():
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n, dtype=np.float32) for n in (30, 50, 40)], indexing="ij")
    vol = 10.0 * (0.5 - np.sqrt(x * x + y * y + z * z)) + 25.0  # surface at level 25
    v, f = surface_nets(vol, 25.0)
    assert f.shape[0] > 500
    counts, _ = edge_stats(f)
    assert (counts == 2).all()
    # vertices come back in (axis0, axis1, axis2) index order
    assert v[:, 0].max() <= 29 and v[:, 1].max() <= 49 and v[:, 2].max() <= 39


def test_padding_closes_shapes_touching_the_boundary():
    x, y, z = grid(32)
    vol = 1.5 - np.sqrt(x * x + y * y + z * z)  # larger than the box: surface is clipped by it
    open_v, open_f = surface_nets(vol, 0.0)
    counts, _ = edge_stats(open_f) if open_f.shape[0] else (np.array([2]), True)
    padded = np.pad(vol, 1, constant_values=-1.0)
    v, f = surface_nets(padded, 0.0)
    c2, ok = edge_stats(f)
    assert (c2 == 2).all() and ok and signed_volume(v, f) > 0


def test_components_and_floater_removal():
    x, y, z = grid(64)
    big = 0.55 - np.sqrt((x + 0.2) ** 2 + y ** 2 + z ** 2)
    speck = 0.06 - np.sqrt((x - 0.8) ** 2 + (y - 0.8) ** 2 + (z - 0.8) ** 2)
    v, f = surface_nets(np.maximum(big, speck), 0.0)
    lab = connected_components(f, v.shape[0])
    assert len(np.unique(lab[np.unique(f)])) == 2
    colors = np.random.rand(v.shape[0], 3).astype(np.float32)
    v2, f2, c2 = keep_large_components(v, f, colors, min_ratio=0.02)
    assert len(np.unique(connected_components(f2, v2.shape[0]))) == 1
    assert c2.shape[0] == v2.shape[0] and f2.max() < v2.shape[0]
    assert f2.shape[0] < f.shape[0]


def test_components_numpy_fallback_matches_scipy(monkeypatch):
    import builtins
    x, y, z = grid(40)
    vol = np.maximum(0.4 - np.sqrt((x + 0.5) ** 2 + y ** 2 + z ** 2), 0.3 - np.sqrt((x - 0.55) ** 2 + y ** 2 + z ** 2))
    v, f = surface_nets(vol, 0.0)
    a = connected_components(f, v.shape[0])
    real_import = builtins.__import__

    def no_scipy(name, *args, **kw):
        if name.startswith("scipy"):
            raise ImportError("blocked")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    b = connected_components(f, v.shape[0])
    monkeypatch.undo()
    used = np.unique(f)
    assert len(np.unique(a[used])) == len(np.unique(b[used])) == 2


def test_taubin_preserves_volume_and_reduces_noise():
    x, y, z = grid(56)
    vol = 0.6 - np.sqrt(x * x + y * y + z * z)
    v, f = surface_nets(vol, 0.0)
    world = v * (2 / 55) - 1
    noisy = world + np.random.RandomState(0).normal(0, 0.004, world.shape).astype(np.float32)
    sm = taubin_smooth(noisy, f, iters=6)
    r_before = np.linalg.norm(noisy, axis=1).std()
    r_after = np.linalg.norm(sm, axis=1).std()
    assert r_after < r_before * 0.7
    assert abs(signed_volume(sm, f) / signed_volume(world, f) - 1) < 0.05


def _smooth_noise(n, seed, blur=2):
    rng = np.random.RandomState(seed)
    v = rng.normal(size=(n, n, n)).astype(np.float32)
    for _ in range(blur):  # cheap box blur -> blobby random field with many saddles
        for ax in range(3):
            v = (np.roll(v, 1, ax) + v + np.roll(v, -1, ax)) / 3.0
    return v


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noisy_field_is_edge_manifold_with_repair(seed):
    vol = np.pad(_smooth_noise(44, seed), 1, constant_values=-1.0)
    v, f = surface_nets(vol, 0.0, manifold=True)
    assert len(f) > 500
    counts, reverse_ok = edge_stats(f)
    assert (counts == 2).all(), f"non-manifold edges left: {(counts != 2).sum()}"
    assert reverse_ok
    assert signed_volume(v, f) != 0


def test_repair_actually_needed_and_touches_very_few_voxels():
    from aura_white.mesh.surface_nets import resolve_ambiguous_faces
    # worst case: pure random blobs (every voxel is near the level, saddles everywhere)
    vol = np.pad(_smooth_noise(44, 0), 1, constant_values=-1.0)
    _, f_raw = surface_nets(vol, 0.0, manifold=False)
    counts_raw, _ = edge_stats(f_raw)
    assert (counts_raw != 2).any(), "test field should contain checkerboard faces"
    assert np.mean((resolve_ambiguous_faces(vol, 0.0) > 0) != (vol > 0)) < 0.01

    # realistic case: a sphere with mild surface noise
    x, y, z = grid(64)
    noisy = (0.6 - np.sqrt(x * x + y * y + z * z)) + 0.02 * _smooth_noise(64, 5)
    fixed = resolve_ambiguous_faces(noisy, 0.0)
    band = np.abs(noisy) < 0.05
    assert np.sum((fixed > 0) != (noisy > 0)) < 0.02 * band.sum()
    v, f = surface_nets(noisy, 0.0)
    counts, ok = edge_stats(f)
    assert (counts == 2).all() and ok
