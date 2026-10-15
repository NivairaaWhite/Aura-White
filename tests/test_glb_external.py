import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from aura_white.backends import glb
from aura_white.backends.base import BackendUnavailable
from aura_white.backends.external import Pixal3DGeometry, Trellis2Geometry
from aura_white.backends.geometry import make_geometry, select_geometry


def test_glb_roundtrip_and_node_transform(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=2)
    p = tmp_path / "a.glb"
    glb.write_glb_mesh(p, m.vertices, m.faces)
    v, f = glb.read_glb(p)
    assert np.allclose(v, m.vertices, atol=1e-6) and np.array_equal(f, m.faces)
    sc = trimesh.Scene()
    sc.add_geometry(m, transform=trimesh.transformations.translation_matrix([1, 2, 3]))
    q = tmp_path / "b.glb"
    sc.export(q)
    v2, f2 = glb.read_glb(q)
    assert np.allclose(v2.mean(0), [1, 2, 3], atol=1e-3) and len(f2) == len(m.faces)


def fake_repo(tmp_path, body):
    repo = tmp_path / "Pixal3D"
    repo.mkdir()
    (repo / "inference.py").write_text(textwrap.dedent(body))
    return repo


def test_subprocess_backend_runs_upstream_and_converts_frame(tmp_path):
    root = Path(__file__).resolve().parents[1]
    repo = fake_repo(tmp_path, f"""
        import argparse, sys
        sys.path.insert(0, {str(root)!r})
        import trimesh
        from aura_white.backends.glb import write_glb_mesh
        ap = argparse.ArgumentParser(); ap.add_argument("--image"); ap.add_argument("--output")
        ap.add_argument("--low_vram", action="store_true"); ap.add_argument("--resolution", type=int)
        a = ap.parse_args()
        m = trimesh.creation.box(extents=(1.0, 2.0, 3.0))       # glTF frame: x wide, y tall, z deep
        write_glb_mesh(a.output, m.vertices, m.faces)
    """)
    g = Pixal3DGeometry(repo=str(repo), python=sys.executable)
    assert g.status()[0]
    res = g.generate(Image.new("RGBA", (16, 16), (255, 0, 0, 255)), seed=3)
    ext = res.verts.max(0) - res.verts.min(0)
    assert np.allclose(ext, [3.0, 1.0, 2.0], atol=1e-5), "glTF (x,y,z) must become canonical (z,x,y)"
    assert res.info["backend"] == "pixal3d"


def test_subprocess_failure_is_a_readable_backend_error(tmp_path):
    repo = fake_repo(tmp_path, "import sys; sys.stderr.write('CUDA out of memory\\n'); sys.exit(3)")
    g = Pixal3DGeometry(repo=str(repo), python=sys.executable)
    with pytest.raises(BackendUnavailable, match="out of memory"):
        g.generate(Image.new("RGBA", (8, 8)))


def test_missing_repo_reports_how_to_fix():
    ok, why = Trellis2Geometry(repo="/nonexistent/x").status()
    assert not ok and "TRELLIS2" in why.upper()


def test_backend_names_and_auto_fall_back():
    assert make_geometry("pixal3d").name == "pixal3d"
    assert make_geometry("trellis2").name == "trellis2"
    assert make_geometry("relief").name == "relief"
    with pytest.raises(ValueError):
        make_geometry("nope")
    assert select_geometry("auto").name in ("pixal3d", "trellis2", "hunyuan3d", "instantmesh", "aura")
