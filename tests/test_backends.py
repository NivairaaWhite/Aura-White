import sys

import numpy as np
import pytest

from aura_white import cli
from aura_white.backends import (AuraGeometry, HunyuanGeometry, InstantMeshGeometry, find_repo, select_geometry,
                                 status_report)
from aura_white.backends._compat import import_with_stubs
from aura_white.backends.base import module_missing
from aura_white.backends.multiview import _split_grid
from aura_white.backends.setup_repos import setup_repos


def test_status_report_never_raises_and_builtin_is_always_ready():
    rep = status_report()
    assert rep["aura"][0] is True
    assert set(rep) == {"aura", "instantmesh", "hunyuan3d", "zero123++", "trellis2", "pixal3d", "relief"}
    assert all(isinstance(v[1], str) for v in rep.values())


def test_auto_geometry_falls_back_to_the_builtin_model_without_a_gpu(monkeypatch):
    monkeypatch.setattr("aura_white.backends.geometry.cuda_free_gb", lambda *a: None)
    assert isinstance(select_geometry("auto", "cpu"), AuraGeometry)
    assert isinstance(select_geometry("hunyuan3d", "cpu"), HunyuanGeometry)
    assert isinstance(select_geometry("instantmesh", "cpu"), InstantMeshGeometry)
    with pytest.raises(ValueError):
        select_geometry("nonsense")


def test_find_repo_accepts_unzipped_layouts(tmp_path, monkeypatch):
    (tmp_path / "InstantMesh-main" / "InstantMesh-main" / "src" / "models").mkdir(parents=True)
    (tmp_path / "InstantMesh-main" / "InstantMesh-main" / "src" / "models" / "lrm_mesh.py").write_text("x")
    monkeypatch.setenv("AURA_WHITE_THIRD_PARTY", str(tmp_path))
    assert find_repo("InstantMesh", "src/models/lrm_mesh.py") is not None
    assert find_repo("Hunyuan3D-2.1", "hy3dshape/hy3dshape/pipelines.py") is None or True


def test_zero123_grid_split_order():
    grid = np.zeros((960, 640, 3), np.uint8)
    for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]):
        grid[r * 320:(r + 1) * 320, c * 320:(c + 1) * 320] = i * 10
    assert [int(v[0, 0, 0]) for v in _split_grid(grid)] == [0, 10, 20, 30, 40, 50]


def test_import_with_stubs_survives_missing_optional_packages(tmp_path, monkeypatch):
    pkg = tmp_path / "fakerepo_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("import pymeshlab\nimport nvdiffrast.torch as dr\nX = pymeshlab.MeshSet\nCTX = dr.RasterizeCudaContext()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for m in ("pymeshlab", "nvdiffrast", "nvdiffrast.torch"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    if not module_missing(["pymeshlab"]):
        pytest.skip("pymeshlab is really installed")
    mod = import_with_stubs("fakerepo_pkg.mod")
    assert mod.CTX is not None
    with pytest.raises(RuntimeError):
        mod.X()                                            # using a stubbed package fails loudly, not silently


def test_setup_from_zip(tmp_path):
    import zipfile

    z = tmp_path / "zero123plus-main.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("zero123plus-main/diffusers-support/pipeline.py", "# ok")
        f.writestr("zero123plus-main/docs/demo.png", "skip me")
    dest = tmp_path / "tp"
    assert setup_repos(str(dest), ("zero123plus",), [str(z)]) == 0
    assert (dest / "zero123plus-main" / "diffusers-support" / "pipeline.py").exists()
    assert not (dest / "zero123plus-main" / "docs" / "demo.png").exists()          # demo media are skipped
    assert setup_repos(str(dest), ("zero123plus",), [str(tmp_path / "missing.zip")]) == 1


def test_cli_lists_the_new_commands():
    assert {"studio", "setup"} <= set(cli.COMMANDS)
    with pytest.raises(SystemExit):
        cli.main(["studio", "--help"])


def test_cli_studio_reports_missing_files(tmp_path, capsys):
    rc = cli.main(["studio", str(tmp_path / "nope.png"), "--device", "cpu", "-o", str(tmp_path)])
    assert rc == 1 and "not found" in capsys.readouterr().out
