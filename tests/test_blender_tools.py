import stat
import sys
import textwrap
from pathlib import Path

import numpy as np

from aura_white.backends import blender_tools as bt


def _fake_blender(tmp_path) -> Path:
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / "blender"
    fake.write_text("#!" + sys.executable + "\n" + textwrap.dedent(f"""
        import sys, argparse
        from PIL import Image
        import numpy as np
        sys.path.insert(0, {str(root)!r})
        from aura_white.backends.glb import read_glb, write_glb_mesh

        if "--version" in sys.argv:
            print("Blender 4.2.3 (hash abcdef1234)")
            sys.exit(0)
        a = sys.argv[sys.argv.index("--") + 1:]
        ap = argparse.ArgumentParser()
        ap.add_argument("--task", required=True)
        ap.add_argument("--in", dest="inp", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--size", type=int, default=1024)
        ap.add_argument("--samples", type=int, default=64)
        ap.add_argument("--target-faces", type=int, default=20000)
        ap.add_argument("--voxel-size", type=float, default=0.01)
        ap.add_argument("--preserve-boundary", action="store_true")
        args = ap.parse_args(a)
        v, f = read_glb(args.inp)
        if args.task == "ao_bake":
            img = np.full((args.size, args.size), 200, np.uint8)
            img[: args.size // 2] = 120
            Image.fromarray(img).save(args.out)
        elif args.task == "quad_remesh":
            write_glb_mesh(args.out, v, f[: max(1, len(f) // 2)])
        elif args.task == "voxel_remesh":
            write_glb_mesh(args.out, v + 0.001, f)
        print("fake blender ok:", args.task)
    """))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR | stat.S_IXUSR)
    return fake


def _mesh():
    rng = np.random.default_rng(0)
    verts = rng.random((20, 3)).astype(np.float32)
    faces = np.array([[i, i + 1, i + 2] for i in range(0, 15, 3)], np.int64)
    uv = rng.random((20, 2)).astype(np.float32)
    return verts, faces, uv


def test_missing_blender_reports_how_to_fix(monkeypatch):
    monkeypatch.delenv("AURA_WHITE_BLENDER", raising=False)
    monkeypatch.setattr(bt, "_candidates", lambda: iter([None]))
    ok, why = bt.status()
    assert not ok and "AURA_WHITE_BLENDER" in why
    assert bt.bake_ao_curvature(*_mesh()[:2], *_mesh()) is None


def test_status_reports_a_found_executable(tmp_path, monkeypatch):
    fake = _fake_blender(tmp_path)
    monkeypatch.setenv("AURA_WHITE_BLENDER", str(fake))
    ok, exe = bt.status()
    assert ok and exe == str(fake)


def test_old_blender_version_is_rejected(tmp_path, monkeypatch):
    fake = tmp_path / "blender"
    fake.write_text("#!" + sys.executable + "\nimport sys\nprint('Blender 3.6.0')\n")
    fake.chmod(0o755)
    monkeypatch.setenv("AURA_WHITE_BLENDER", str(fake))
    ok, why = bt.status()
    assert not ok and "3.6" in why


def test_ao_bake_round_trips_through_the_subprocess(tmp_path, monkeypatch):
    fake = _fake_blender(tmp_path)
    monkeypatch.setenv("AURA_WHITE_BLENDER", str(fake))
    v, f, uv = _mesh()
    img = bt.bake_ao_curvature(v, f, v, f, uv, size=64)
    assert img is not None and img.shape == (64, 64) and img.dtype == np.uint8
    assert img[0, 0] == 120 and img[-1, -1] == 200


def test_quad_remesh_round_trips_and_reduces_faces(tmp_path, monkeypatch):
    fake = _fake_blender(tmp_path)
    monkeypatch.setenv("AURA_WHITE_BLENDER", str(fake))
    v, f, _ = _mesh()
    out = bt.quad_remesh(v, f, target_faces=2)
    assert out is not None
    v2, f2 = out
    assert len(f2) < len(f) and v2.shape[1] == 3


def test_voxel_remesh_round_trips(tmp_path, monkeypatch):
    fake = _fake_blender(tmp_path)
    monkeypatch.setenv("AURA_WHITE_BLENDER", str(fake))
    v, f, _ = _mesh()
    out = bt.voxel_remesh(v, f, voxel_size=0.02)
    assert out is not None
    v2, f2 = out
    assert not np.allclose(v2, v) and len(f2) == len(f)


def test_subprocess_failure_returns_none_not_an_exception(tmp_path, monkeypatch):
    fake = tmp_path / "blender"
    fake.write_text("#!" + sys.executable + "\nimport sys\n"
                    "print('Blender 4.2.0') if '--version' in sys.argv else sys.exit(1)\n")
    fake.chmod(0o755)
    monkeypatch.setenv("AURA_WHITE_BLENDER", str(fake))
    v, f, uv = _mesh()
    messages = []
    assert bt.bake_ao_curvature(v, f, v, f, uv, progress=messages.append) is None
    assert any("failed" in m for m in messages)
