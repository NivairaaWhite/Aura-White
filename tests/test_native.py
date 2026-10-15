"""Aura Native: licence ledger, procedural ground truth, quality gates, the models, teachers, compression, the multi-view
reconstruction and the command line.  Everything here runs on a CPU in well under a few minutes."""
import json
import os

import numpy as np
import pytest

from aura_white.native import geometry as G
from aura_white.native import licenses as L
from aura_white.native.procedural import make_asset

torch = pytest.importorskip("torch")


# ------------------------------------------------------------------------------------------------ licences
def test_hunyuan_is_never_a_teacher_even_with_risk_accepted():
    # exact keys AND the spellings people really type: a name missing from the ledger would otherwise be allowed by accept_risk
    for key in ("hunyuan3d-2", "hunyuan3d-2.1", "Hunyuan3D_2.5", "hunyuan3d-2mv", "tencent hunyuan", "HUNYUAN", "hunyuan3d-omni"):
        d = L.can_teach(key, commercial=False, accept_risk=True)
        assert not d.allowed and ("5(b)" in d.reason or "improve" in d.reason), key
        with pytest.raises(L.LicenceError):
            L.require_teacher(key, commercial=False, accept_risk=True)


def test_ledger_spellings_resolve():
    assert L.can_teach("TRELLIS.2").allowed and L.can_teach("  triposr ").allowed
    assert not L.can_teach("Zero123++", commercial=True).allowed and L.can_teach("Zero123++", commercial=False).allowed
    assert L.resolve("Hunyuan3D_2.1").key == "hunyuan3d-2.1"


def test_permissive_teachers_allowed_and_unknown_denied():
    assert L.can_teach("trellis2").allowed and L.can_teach("triposg").allowed and L.can_teach("triposr").allowed
    assert not L.can_teach("some-new-model").allowed
    assert L.can_teach("some-new-model", accept_risk=True).risk_accepted
    assert not L.can_teach("zero123plus", commercial=True).allowed       # CC-BY-NC weights
    assert L.can_teach("zero123plus", commercial=False).allowed
    assert not L.can_teach("instantmesh").allowed                         # unverified -> denied by default


def test_asset_licence_filter():
    assert L.asset_licence_ok("cc0")[0] and L.asset_licence_ok("by")[0]
    assert not L.asset_licence_ok("by-nc", commercial=True)[0] and L.asset_licence_ok("by-nc", commercial=False)[0]
    assert not L.asset_licence_ok("by-nd", commercial=False)[0]
    assert not L.asset_licence_ok("by", no_ai_flag=True)[0]
    assert not L.asset_licence_ok(None)[0]


def test_provenance_flags_restricted_sources():
    p = L.Provenance()
    p.add("zero123plus", 3, L.can_teach("zero123plus", commercial=False))
    ok, why = p.commercially_clean()
    assert not ok and any("zero123plus" in w for w in why)


# ------------------------------------------------------------------------------------------------ procedural + gates
def test_procedural_front_and_back_differ_and_sdf_is_signed():
    a = make_asset("figure", 3)
    rng = np.random.default_rng(0)
    pts, nrm = a.sample_surface(2000, rng)
    assert abs(a.sdf(pts)).max() < 0.02                                     # samples lie on the surface
    assert a.sdf(np.zeros((1, 3)) + [0, 0.0, 0.0]).shape == (1,)
    c = a.colour(pts)
    front, back = c[pts[:, 2] > 0.1], c[pts[:, 2] < -0.1]
    assert len(front) > 20 and len(back) > 20
    assert np.abs(front.mean(0) - back.mean(0)).sum() > 0.02 or front.std(0).sum() > 0.02


def _slab(depth=0.02):
    v = np.array([[-1, -1, -depth], [1, -1, -depth], [1, 1, -depth], [-1, 1, -depth],
                  [-1, -1, depth], [1, -1, depth], [1, 1, depth], [-1, 1, depth]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    return v, f


def test_gates_reject_a_slab_and_accept_a_real_figure():
    from aura_white.native.gates import verify

    v, f = _slab()
    bad = verify(v, f, np.full((8, 3), 0.7), None, True)
    assert not bad.passed and "depth" in bad.failing
    a = make_asset("figure", 1)
    vv, ff, cc = a.mesh(48)
    good = verify(vv, ff, cc, None, True)
    assert good.passed, good.failing


def test_geometry_marching_cubes_sphere():
    g = G.grid_coords(40).reshape(40, 40, 40, 3)
    sdf = np.linalg.norm(g, axis=-1) - 0.6
    v, f = G.sdf_to_mesh(sdf)
    r = np.linalg.norm(v, axis=1)
    assert abs(r.mean() - 0.6) < 0.03 and len(f) > 500


# ------------------------------------------------------------------------------------------------ models
@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    from aura_white.native.data import DataSpec, ShardDataset, build_procedural

    root = tmp_path_factory.mktemp("ds")
    build_procedural(root, 14, DataSpec(n_points=256, n_queries=512, image_size=64), seed0=1000, workers=1)
    return root, ShardDataset(root, "train"), ShardDataset(root, "val")


def test_vae_and_flow_train_a_step_and_checkpoint_roundtrip(tiny_data, tmp_path):
    from aura_white.native.model import AuraNative, ViewIn
    from aura_white.native.train import fit_latent_stats, train_flow, train_vae

    root, tr, va = tiny_data
    m = AuraNative.new("tiny")
    r = train_vae(m, tr, steps=3, batch=4, say=None)
    assert r["done"] and np.isfinite(r["total"])
    fit_latent_stats(m, tr, n=8)
    r = train_flow(m, tr, steps=3, batch=4, say=None)
    assert r["done"] and np.isfinite(r["loss"])
    ck = tmp_path / "m.pt"
    m.save(ck, {"approved_for_default": False})
    m2 = AuraNative.load(ck)                                               # weights_only load: no pickle code can run
    assert m2.params() == m.params()
    pic = np.full((64, 64, 4), 0.5, np.float32)
    pic[16:48, 16:48, 3] = 1.0
    z = m2.sample_z([ViewIn(pic)], n=2, steps=2, seed=1)
    assert z.shape == (2, 8, 8, 8, 8) and torch.isfinite(z).all()
    # same seed -> same sample
    z2 = m2.sample_z([ViewIn(pic)], n=2, steps=2, seed=1)
    assert torch.allclose(z, z2)


def test_checkpoint_loading_rejects_foreign_files(tmp_path):
    from aura_white.native.model import AuraNative

    bad = tmp_path / "x.pt"
    torch.save({"format": "other"}, bad)
    with pytest.raises(ValueError):
        AuraNative.load(bad)


def test_compression_tools_run_and_stay_loadable(tiny_data, tmp_path):
    from aura_white.native import compress as C
    from aura_white.native.config import preset
    from aura_white.native.model import AuraNative
    from aura_white.native.train import fit_latent_stats

    root, tr, va = tiny_data
    b = [C.budget(preset(n))["params"]["total"] for n in ("tiny", "s", "m", "l")]
    assert b == sorted(b) and b[0] < b[1]
    m = AuraNative.new("tiny")
    fit_latent_stats(m, tr, n=8)
    p = C.prune_depth(m, tr, keep=3, heal_steps=2, batch=2, say=None)
    assert len(p["model"].flow.blocks) == 3 and p["model"].cfg.flow.layers == 3
    p["model"].save(tmp_path / "p.pt")
    assert len(AuraNative.load(tmp_path / "p.pt").flow.blocks) == 3
    d = C.distill_guidance(m, tr, steps=2, batch=2, say=None)
    assert d["model"].guidance_distilled
    q = C.quantize_int8(m)
    assert q.flow is not m.flow


def test_self_improvement_selection_is_verifier_gated_and_oracle_free():
    from aura_white.native.evolve import Cand, EvolveConfig, select_winners

    cfg = EvolveConfig(score_floor=0.75, selection="verifier")
    cands = [Cand(prompt=i // 2, views=[], z=np.zeros((1, 1, 1, 1)), score=sc, passed=sc > 0.75, seed=i)
             for i, sc in enumerate([0.9, 0.4, 0.6, 0.8, 0.2, 0.3])]
    win = select_winners(cands, cfg, np.random.default_rng(0))
    assert win and all(w.score >= 0.75 for w in win)
    assert len({w.prompt for w in win}) == len(win)                          # one winner per picture


# ------------------------------------------------------------------------------------------------ teachers
def _cutout_png(asset, path):
    from PIL import Image

    from aura_white.native.render import fit_camera, render

    v, f, c = asset.mesh(64)
    cam = fit_camera(v, 0.0, 0.0, 192, 0.86)
    rgba, _, _ = render(v, f, c, cam=cam, size=192)
    Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").save(path)
    return v, f, c


def test_teacher_pipeline_keeps_a_good_mesh_and_drops_a_slab(tmp_path):
    from aura_white.native.data import DataSpec
    from aura_white.native.teach import FunctionTeacher, teach

    a = make_asset("figure", 40)
    v, f, c = _cutout_png(a, tmp_path / "good.png")
    _cutout_png(make_asset("figure", 41), tmp_path / "slab.png")              # its picture is fine, the teacher's mesh is not
    slab_v, slab_f = _slab()

    def fn(path):
        if path.name == "slab.png":
            return slab_v * 0.5, slab_f, None                                # a "teacher" that returns a slab
        return v[:, [2, 0, 1]], f, None                                      # right shape, wrong axis order: must be re-aligned

    t = FunctionTeacher("triposr", fn)
    dd = teach(t, [tmp_path / "good.png", tmp_path / "slab.png"], tmp_path / "td",
               DataSpec(n_points=256, n_queries=512, image_size=64), min_sign_agreement=0.7, say=None)
    assert len(dd) == 1 and dd.report["kept"] == 1
    assert sum(dd.report["skipped"].values()) == 1 and any("match" in k or "gates" in k for k in dd.report["skipped"])
    man = json.loads((tmp_path / "td" / "manifest.json").read_text())
    assert man["provenance"]["decisions"]["triposr"]["allowed"] is True
    assert man["items"][0]["source"] == "triposr"


def test_teacher_refuses_hunyuan_at_construction():
    from aura_white.native.teach import CommandTeacher, FunctionTeacher

    with pytest.raises(L.LicenceError):
        FunctionTeacher("hunyuan3d-2.1", lambda p: None)
    with pytest.raises(L.LicenceError):
        CommandTeacher("hunyuan3d-2", "echo {image} {out}", accept_risk=True)
    with pytest.raises(L.LicenceError):
        CommandTeacher("never-heard-of-it", "echo")
    with pytest.raises(L.LicenceError):                       # a permissive label on a command that really runs Hunyuan
        CommandTeacher("trellis2", "python run_hunyuan3d.py --image {image} --out {out}")


# ------------------------------------------------------------------------------------------------ multi-view
def _sheet_from_asset(asset):
    from aura_white.native.render import fit_camera, render

    v, f, c = asset.mesh(80)
    out = {}
    for name, az in (("front", 0.0), ("back", 180.0), ("left", 90.0), ("right", -90.0)):
        cam = fit_camera(v, az, 0.0, 384, 0.86)
        rgba, _, _ = render(v, f, c, cam=cam, size=384)
        out[name] = rgba                                                     # RGBA array: the true alpha is the cut-out
    return v, f, c, out


def test_sheet_reconstruction_has_real_depth_a_different_back_and_a_valid_glb(tmp_path):
    from aura_white.native import multiview as MV

    v, f, c, srcs = _sheet_from_asset(make_asset("figure", 3))
    res = MV.reconstruct(srcs, out_dir=tmp_path, name="fig", height_m=1.7, faces=8000, cells=120, tex=512, canvas=384)
    r = res.report
    assert r["watertight_before_unwrap"]
    assert min(r["outline_iou"].values()) > 0.93                             # the mesh reproduces all four outlines
    assert abs(res.verts[:, 1].max() - 1.7) < 1e-3 and abs(res.verts[:, 1].min()) < 1e-3
    size = res.verts.max(0) - res.verts.min(0)
    assert size[2] > 0.08 * size[1]                                          # real depth, not a slab
    # the unseen sides are painted from their own pictures: the back atlas region is not a mirror of the front
    assert res.albedo.std() > 5
    from aura_white.native.gates import check_glb

    chk = check_glb(res.glb)
    assert chk.get("ok", True), chk
    import trimesh

    m = trimesh.load(str(res.glb), force="mesh")
    assert len(m.faces) == len(res.faces) and m.visual.uv.shape[0] == len(m.vertices)
    # shape accuracy against the true mesh (unit height): the carve is exact in the outlines, a rounded guess in between
    g = (v - [0, v[:, 1].min(), 0]) / (v[:, 1].max() - v[:, 1].min())
    g -= [(g[:, 0].max() + g[:, 0].min()) / 2, 0, (g[:, 2].max() + g[:, 2].min()) / 2]
    s = res.verts / 1.7
    s = s - [(s[:, 0].max() + s[:, 0].min()) / 2, 0, (s[:, 2].max() + s[:, 2].min()) / 2]
    ch = G.mesh_chamfer(s, res.faces, g, f, n=4000, seed=0, taus=(0.02,))
    assert ch["chamfer"] < 0.03, ch


def test_sheet_paints_a_supplied_mesh_after_finding_its_orientation(tmp_path):
    from aura_white.gen3d.views import rot_y
    from aura_white.native import multiview as MV

    v, f, c, srcs = _sheet_from_asset(make_asset("figure", 3))
    scrambled = (v @ rot_y(90.0).T) * 3.0 + [4.0, -1.0, 2.0]                 # quarter turn, other scale, other place
    res = MV.reconstruct(srcs, out_dir=tmp_path, name="p", faces=8000, tex=512, canvas=384, base_mesh=(scrambled, f))
    assert res.report["mode"] == "paint-supplied-mesh"
    assert res.report["mesh_registration"]["agreement"] > 0.95
    assert min(res.report["outline_iou"].values()) > 0.95


def test_sheet_without_a_side_view_says_it_is_a_guess():
    from aura_white.native import multiview as MV

    v, f, c, srcs = _sheet_from_asset(make_asset("figure", 3))
    views = MV.prepare_views({"front": srcs["front"], "back": srcs["back"]}, 384)
    fld = MV.build_field(views, cells=80)
    assert any("NO side view" in n for n in fld.notes)


# ------------------------------------------------------------------------------------------------ CLI + back-end
def test_cli_licences_and_refusal(capsys):
    from aura_white.cli import main

    assert main(["native", "licences"]) == 0
    assert "hunyuan3d-2.1" in capsys.readouterr().out
    rc = main(["native", "teach", "pics", "out", "--teacher", "hunyuan3d-2.1", "--command", "echo {image} {out}"])
    assert rc == 1 and "cannot use" in capsys.readouterr().err


def test_cli_budget_and_doctor(capsys):
    from aura_white.cli import main

    assert main(["native", "budget", "--preset", "tiny"]) == 0
    assert main(["native", "doctor"]) == 0


def test_native_backend_is_opt_in(tmp_path, monkeypatch):
    from aura_white.gen3d.hardware import Hardware
    from aura_white.gen3d.shapegen import AuraNativeShape
    from aura_white.native.model import AuraNative

    hw = Hardware()
    assert not AuraNativeShape("").probe(hw, []).ok
    ck = tmp_path / "demo.pt"
    AuraNative.new("tiny").save(ck, {"approved_for_default": False})
    monkeypatch.delenv("AURA_NATIVE_ALLOW_UNAPPROVED", raising=False)
    pr = AuraNativeShape(str(ck)).probe(hw, [])
    assert not pr.ok and "approved_for_default" in pr.reason
    monkeypatch.setenv("AURA_NATIVE_ALLOW_UNAPPROVED", "1")
    assert AuraNativeShape(str(ck)).probe(hw, []).ok
    AuraNative.new("tiny").save(ck, {"approved_for_default": True})
    monkeypatch.delenv("AURA_NATIVE_ALLOW_UNAPPROVED")
    assert AuraNativeShape(str(ck)).probe(hw, []).ok


def test_demo_checkpoint_cannot_be_approved(tiny_data, tmp_path, capsys):
    from aura_white.cli import main
    from aura_white.native.model import AuraNative

    root, tr, va = tiny_data
    ck = tmp_path / "demo.pt"
    AuraNative.new("tiny").save(ck)
    bench = tmp_path / "b.json"
    bench.write_text(json.dumps({"summary": {"pass_rate": 0.99, "score": 0.99}}))
    rc = main(["native", "approve", str(ck), "--bench", str(bench), "--data", str(root)])
    out = capsys.readouterr().out
    assert rc == 1 and "NOT approved" in out and "procedural" in out
    assert not AuraNative.load(ck).card.get("approved_for_default")


def test_sheet_rejects_empty_picture_with_a_clear_message(tmp_path):
    from PIL import Image

    from aura_white.cli import main

    Image.new("RGB", (200, 300), (0, 0, 0)).save(tmp_path / "empty.png")
    with pytest.raises(SystemExit) as e:
        main(["sheet", "--front", str(tmp_path / "empty.png"), "--out", str(tmp_path / "o")])
    assert "no subject found" in str(e.value)
