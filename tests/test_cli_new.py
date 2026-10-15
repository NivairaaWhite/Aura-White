import json

import numpy as np
import pytest
import trimesh

from aura_white.backends.glb import read_glb, write_glb_mesh
from aura_white.cli import main


def test_forge_cli_writes_a_glb_with_chains(tmp_path):
    from synth import make_artwork, make_object

    from aura_white.raster import get_raster

    v, f = make_object(2)
    art = make_artwork(get_raster("cpu"), v, f, size=512)
    img_path = tmp_path / "art.png"
    art.save(img_path)
    out = tmp_path / "out" / "relief.glb"
    rc = main(["forge", str(img_path), "-o", str(out)])
    assert rc == 0 and out.exists()
    vv, ff = read_glb(out)
    assert len(vv) > 0 and len(ff) > 0


def test_repair_cli_fixes_a_broken_mesh(tmp_path, capsys):
    m = trimesh.creation.icosphere(subdivisions=3)
    F = np.delete(np.asarray(m.faces), [1, 2, 3], axis=0)
    p = tmp_path / "broken.glb"
    write_glb_mesh(p, m.vertices, F)
    rc = main(["repair", str(p)])
    assert rc == 0
    out = tmp_path / "broken_repaired.glb"
    assert out.exists()
    vv, ff = read_glb(out)
    rep = trimesh.Trimesh(vv, ff, process=False)
    assert rep.is_watertight
    printed = capsys.readouterr().out
    assert "holes_filled" in printed


def test_eval_cli_reports_metrics(tmp_path, capsys):
    m = trimesh.creation.icosphere(subdivisions=2)
    (tmp_path / "pred").mkdir()
    (tmp_path / "gt").mkdir()
    write_glb_mesh(tmp_path / "pred" / "x.glb", m.vertices, m.faces)
    write_glb_mesh(tmp_path / "gt" / "x.glb", m.vertices, m.faces)
    rc = main(["eval", "--pred", str(tmp_path / "pred"), "--gt", str(tmp_path / "gt")])
    assert rc == 0
    out = json.loads((tmp_path / "pred" / "eval_report.json").read_text())
    assert out["n_pairs"] == 1
    assert "f_score=" in capsys.readouterr().out


def test_shape_train_and_complete_cli(tmp_path, capsys):
    ckpt = tmp_path / "shape.pt"
    rc = main(["shape-train", "--steps", "60", "--res", "12", "--patch", "6", "--d-model", "32", "--layers", "2",
              "--heads", "4", "--batch", "8", "--out", str(ckpt)])
    assert rc == 0 and ckpt.exists()
    m = trimesh.creation.icosphere(subdivisions=2)
    keep = np.asarray(m.vertices)[np.asarray(m.faces)].mean(1)[:, 0] > 0
    mesh_path = tmp_path / "half.glb"
    write_glb_mesh(mesh_path, m.vertices, np.asarray(m.faces)[keep])
    rc = main(["shape-complete", str(mesh_path), "--ckpt", str(ckpt), "--steps", "6", "--device", "cpu"])
    assert rc == 0
    assert (tmp_path / "half_completed.glb").exists()


def test_help_exits_via_argparse():
    with pytest.raises(SystemExit):
        main(["--help"])


def test_nonexistent_image_is_a_clean_error_not_a_crash(capsys):
    rc = main(["nonexistent-file.png"])          # no known command -> treated as `generate <image>`
    assert rc == 1
    assert "error:" in capsys.readouterr().err
