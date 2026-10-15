import numpy as np
import pytest
import torch
import trimesh

from aura_white import _isolate
from aura_white.raster import get_raster
from aura_white.texture import decimate, unwrap
import importlib

unwrap_mod = importlib.import_module("aura_white.texture.unwrap")
from aura_white.texture.decimate import cluster_decimate


def meshes():
    return {"sphere": trimesh.creation.icosphere(3), "box": trimesh.creation.box(),
            "torus": trimesh.creation.torus(1.0, 0.35, major_sections=32, minor_sections=16)}


@pytest.mark.parametrize("name", ["sphere", "box", "torus"])
def test_builtin_unwrap_is_valid(name):
    m = meshes()[name]
    v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces)
    nv, nf, uv, vm, used = unwrap(v, f, 512, "builtin")
    assert used == "builtin" and len(nf) == len(f)
    assert uv.min() >= 0 and uv.max() <= 1
    assert np.allclose(nv, v[vm]) and np.allclose(nv[nf], v[f])          # geometry untouched, only split
    out = get_raster("cpu").rasterize_uv(torch.tensor(uv), torch.tensor(nf), 512)
    d1, d2 = uv[nf[:, 1]] - uv[nf[:, 0]], uv[nf[:, 2]] - uv[nf[:, 0]]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).sum()
    assert (out.tri >= 0).float().mean().item() == pytest.approx(area, rel=0.03)   # no overlapping charts


def test_isolation_survives_a_native_crash():
    assert _isolate.call_isolated("aura_white._isolate:selftest_ok", {"x": 21})["y"] == 42
    with pytest.raises(_isolate.IsolatedCallError):
        _isolate.call_isolated("aura_white._isolate:selftest_crash", {})


def test_unwrap_falls_back_when_xatlas_dies(monkeypatch):
    monkeypatch.setattr(unwrap_mod, "have_xatlas", lambda: True)
    monkeypatch.setattr(unwrap_mod, "_XATLAS_BROKEN", False)

    def boom(*a, **k):
        raise _isolate.IsolatedCallError("crashed")

    monkeypatch.setattr(unwrap_mod, "_xatlas_isolated", boom)
    m = trimesh.creation.icosphere(2)
    *_, used = unwrap(np.asarray(m.vertices, np.float32), np.asarray(m.faces), 256, "auto")
    assert used == "builtin"
    with pytest.raises(Exception):
        unwrap(np.asarray(m.vertices, np.float32), np.asarray(m.faces), 256, "xatlas")


@pytest.mark.parametrize("backend", ["auto", "builtin"])
def test_decimate_keeps_shape(backend):
    m = trimesh.creation.icosphere(5)
    v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces)
    if backend == "builtin":
        nv, nf = cluster_decimate(v, f, 3000)
    else:
        nv, nf, _ = decimate(v, f, 3000)
    assert 0.5 * 3000 < len(nf) < 1.6 * 3000
    assert np.abs(np.linalg.norm(nv, axis=1) - 1).max() < 0.02
    vol = np.einsum("ij,ij->i", nv[nf[:, 0]], np.cross(nv[nf[:, 1]], nv[nf[:, 2]])).sum() / 6
    assert vol == pytest.approx(4 / 3 * np.pi, rel=0.03)                 # winding preserved (positive volume)


def test_cluster_decimation_keeps_thin_blades():
    m = trimesh.creation.box((2.0, 0.02, 0.3))
    for _ in range(4):
        m = m.subdivide()
    v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces)
    nv, nf = cluster_decimate(v, f, 1200)
    assert np.allclose(np.ptp(nv, 0), np.ptp(v, 0), rtol=0.05)


def test_decimate_noop_when_small():
    m = trimesh.creation.icosphere(1)
    v, f, used = decimate(np.asarray(m.vertices, np.float32), np.asarray(m.faces), 10_000)
    assert used == "none" and len(f) == len(m.faces)
