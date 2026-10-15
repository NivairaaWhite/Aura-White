import numpy as np
import trimesh

from aura_white.mesh import lod


def _sphere():
    m = trimesh.creation.icosphere(subdivisions=4)
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


def test_lods_are_independent_not_chained_and_shrink_monotonically():
    v, f = _sphere()
    levels = lod.generate_lods(v, f, levels=(1.0, 0.35, 0.12, 0.04))
    assert len(levels) == 4
    assert levels[0].faces_actual == len(f)
    counts = [l.faces_actual for l in levels]
    assert counts == sorted(counts, reverse=True)
    for l in levels:
        assert l.watertight


def test_lods_preserve_roughly_the_original_volume():
    v, f = _sphere()
    levels = lod.generate_lods(v, f, levels=(1.0, 0.2))
    true_vol = trimesh.Trimesh(v, f, process=False).volume
    lod_vol = trimesh.Trimesh(levels[1].verts, levels[1].faces, process=False).volume
    assert abs(lod_vol - true_vol) / true_vol < 0.25


def test_convex_hull_is_closed_and_conservative_for_a_concave_shape():
    a = trimesh.creation.box(extents=(2, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 2))
    b.apply_translation([0.5, 0, 1.5])
    cat = trimesh.util.concatenate([a, b])
    hull = lod.convex_hull_collision(np.asarray(cat.vertices))
    th = trimesh.Trimesh(hull.verts, hull.faces, process=False)
    assert th.is_watertight and th.is_winding_consistent
    assert hull.volume >= cat.volume - 1e-6
    assert hull.kind == "convex_hull"


def test_convex_hull_matches_a_sphere_closely():
    v, f = _sphere()
    hull = lod.convex_hull_collision(v)
    assert abs(hull.volume - trimesh.Trimesh(v, f, process=False).volume) < 0.05


def test_voxel_collision_falls_back_to_none_without_blender(monkeypatch):
    monkeypatch.delenv("AURA_WHITE_BLENDER", raising=False)
    monkeypatch.setattr("aura_white.backends.blender_tools._candidates", lambda: iter([None]))
    v, f = _sphere()
    assert lod.voxel_simplified_collision(v, f, voxel_size=0.1) is None
