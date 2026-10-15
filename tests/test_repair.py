import numpy as np
import pytest
import trimesh

from aura_white.mesh import repair


def sphere():
    m = trimesh.creation.icosphere(subdivisions=3)
    return np.asarray(m.vertices), np.asarray(m.faces).copy()


def test_holes_dupes_flips_are_fixed():
    V, F = sphere()
    F = np.delete(F, [5, 6, 7, 50, 51, 200], axis=0)
    F = np.concatenate([F, F[:10], np.array([[0, 0, 1], [3, 3, 3]])], 0)
    F[100:140] = F[100:140][:, [0, 2, 1]]
    v, f, rep = repair.repair_mesh(V, F)
    assert rep["watertight"] and rep["faces_flipped"] >= 40
    assert trimesh.Trimesh(v, f, process=False).volume == pytest.approx(4.15, abs=0.1)


def test_nonmanifold_edge_is_cut():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [1, 1, 1.0]])
    F = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4], [1, 0, 5]])
    _, _, rep = repair.repair_mesh(V, F, repair.RepairOptions(fill_holes=False, remove_islands=False))
    assert rep["before"]["nonmanifold_edges"] == 1 and rep["after"]["nonmanifold_edges"] == 0


def test_bowtie_vertex_is_split():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    F = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 1]])
    v, f, rep = repair.repair_mesh(V, F, repair.RepairOptions(fill_holes=False, remove_islands=False))
    assert rep["vertices_split"] >= 1
    assert len(v) > 6


def test_triangle_soup_is_welded():
    m = trimesh.creation.icosphere(subdivisions=2)
    sv = np.asarray(m.vertices)[np.asarray(m.faces).ravel()]
    v, f, rep = repair.repair_mesh(sv, np.arange(len(sv)).reshape(-1, 3))
    assert rep["watertight"] and len(v) == 162


def test_dust_is_removed():
    a = trimesh.creation.icosphere(subdivisions=2)
    b = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    b.apply_translation([3, 0, 0])
    cat = trimesh.util.concatenate([a, b])
    v, f, rep = repair.repair_mesh(np.asarray(cat.vertices), np.asarray(cat.faces))
    assert rep["islands_removed"] == 1 and len(f) == len(a.faces)


def test_ear_clip_fills_a_ring_hole():
    m = trimesh.creation.icosphere(subdivisions=2)
    F = np.asarray(m.faces)
    keep = np.linalg.norm(m.vertices[F].mean(1) - np.array([0, 0, 1.0]), axis=1) > 0.5
    v, f, rep = repair.repair_mesh(np.asarray(m.vertices), F[keep])
    assert rep["holes_filled"] >= 1 and rep["watertight"]
