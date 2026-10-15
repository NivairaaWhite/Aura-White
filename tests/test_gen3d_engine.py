"""End to end: pictures -> engine -> asset pack, with a fake shape generator that misbehaves on purpose (the neural
models themselves are tested on your machine with `aura-white doctor3d --deep`)."""
import json

import numpy as np
import pytest
from PIL import Image

import prims
from gen3d_fakes import TRUTH, FakeShape, install_fake, photo
from aura_white.asset.meshio import read_gltf
from aura_white.asset.pipeline import AssetOptions, build_asset
from aura_white.gen3d import engine as E
from aura_white.gen3d import paint as PT
from aura_white.gen3d import shapegen as S
from aura_white.gen3d import views as V
from aura_white.gen3d.audit import audit_mesh
from aura_white.gen3d.recipe import Recipe

REC = Recipe.preset("standard").with_(candidates=2, max_rounds=2, fusion_grid=64, paint_views=False)
ASSET = dict(lods=2, preview=False, atlas=512)


def opts(**kw):
    return E.MakeOptions(recipe=REC, asset=ASSET, **kw)


def test_one_picture_to_a_closed_solid_asset(tmp_path, monkeypatch):
    fake = install_fake(monkeypatch)
    photo(0).save(tmp_path / "knight.png")
    res = E.Aura3D(opts(), lambda s: None).make([str(tmp_path / "knight.png")], tmp_path / "out")
    assert res.passed and res.version == "v1"
    best = res.summary["best"]
    assert best["depth_ratio"] > 0.4 and best["profile"] == "solid" and best["flat_risk"] == 0
    assert len(fake.calls) >= 2 and fake.calls[0][:2] == (0, 0)             # the flat first candidate was generated, then rejected
    assert (res.out_dir / "job.json").is_file() and (res.out_dir / "report.md").is_file()
    assert res.pack and (res.pack / "game.json").is_file()
    glb = next(res.pack.glob("*.glb"))
    v, f = read_gltf(str(glb))[:2]
    assert audit_mesh(np.asarray(v, float), np.asarray(f, int), expect_solid=False).depth_ratio > 0.3


def test_the_job_records_the_quality_signature_and_the_outline_previews(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    photo(0).save(tmp_path / "x.png")
    res = E.Aura3D(opts(), lambda s: None).make([str(tmp_path / "x.png")], tmp_path / "out", text=None)
    job = json.loads((res.out_dir / "job.json").read_text())
    assert job["recipe"]["signature"] == REC.signature() == job["versions"][0]["recipe_signature"]
    assert (res.out_dir / "views" / "outline_front.png").is_file()


def test_several_pictures_and_refining_with_a_new_view_keeps_every_version(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    for name, az in (("hero_front.png", 0), ("hero_back.png", 180), ("hero_left.png", 90)):
        photo(az).save(tmp_path / name)
    eng = E.Aura3D(opts(seed=5), lambda s: None)
    r1 = eng.make([str(tmp_path / n) for n in ("hero_front.png", "hero_back.png", "hero_left.png")], tmp_path / "out")
    assert r1.passed and set(r1.summary["best"]["per_view"]) == {"front", "left", "back"}
    assert {e["label"] for e in r1.summary["extra_views"]} == {"left", "back"}          # real photos are projected into the texture
    photo(-90).save(tmp_path / "right_side.png")
    r2 = E.Aura3D(opts(), lambda s: None).refine(r1.out_dir, add=[f"right={tmp_path / 'right_side.png'}"], feedback="right side wrong")
    assert r2.version == "v2" and "right" in r2.summary["best"]["per_view"]
    job = json.loads((r1.out_dir / "job.json").read_text())
    assert [v["id"] for v in job["versions"]] == ["v1", "v2"] and job["feedback"] == ["right side wrong"]
    assert (r1.out_dir / "v1" / "mesh.glb").is_file() and (r1.out_dir / "v2" / "mesh.glb").is_file()


def test_a_picture_added_in_refine_replaces_the_stored_one_for_that_view(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    photo(0).save(tmp_path / "front.png")
    photo(180, tint=(40, 190, 70)).save(tmp_path / "back.png")
    r1 = E.Aura3D(opts(), lambda s: None).make([str(tmp_path / "front.png"), str(tmp_path / "back.png")], tmp_path / "out")
    photo(180, tint=(250, 10, 10)).save(tmp_path / "better_back.png")
    r2 = E.Aura3D(opts(), lambda s: None).refine(r1.out_dir, add=[f"back={tmp_path / 'better_back.png'}"])
    stored = np.asarray(Image.open(r1.out_dir / "inputs" / "back.png").convert("RGB"))
    a = np.asarray(Image.open(r1.out_dir / "inputs" / "back.png").convert("RGBA"))[..., 3] > 128
    assert stored[a][:, 0].mean() > stored[a][:, 1].mean()                           # now red, not green
    assert r2.summary["best"]["per_view"].keys() == {"front", "back"}


def test_a_picture_with_no_trustworthy_outline_stops_the_run(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(tmp_path / "noise.png")
    with pytest.raises(Exception) as e:
        E.Aura3D(opts(), lambda s: None).make([str(tmp_path / "noise.png")], tmp_path / "out")
    assert "trustworthy outline" in str(e.value)


def test_with_no_generator_installed_nothing_is_built(tmp_path):
    photo(0).save(tmp_path / "k.png")
    with pytest.raises(S.NoBackend):
        E.Aura3D(opts(), lambda s: None).make([str(tmp_path / "k.png")], tmp_path / "out")


# ------------------------------------------------------------------------------------------ texture projection
def _coloured_view(az, colour, S_=384):
    v, f = TRUTH
    c = V.ViewCam(az, 20.0, S_, S_ * 0.33, 0.0, 0.2)
    m, _, tri, _ = V.render_maps(v, f, c)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    sh = 0.45 + 0.55 * np.clip(fn @ PT.camera_dir(az, 20.0), 0, 1)
    img = np.zeros((S_, S_, 3), np.uint8)
    img[m] = np.clip(sh[tri[m]][:, None] * np.array(colour, float), 0, 255).astype(np.uint8)
    return PT.ExtraView({90: "left", 180: "back"}[az], float(az), 20.0, img, m, 1.0, "user")


def test_extra_views_recolour_their_own_side_and_leave_the_picture_exact(tmp_path, monkeypatch):
    v, f = TRUTH
    v = v.copy()
    v[:, 1] -= v[:, 1].min()
    from aura_white.asset.camera import OrthoCamera, render_view
    cam = OrthoCamera(200, 260, 20.0)
    w = v * 150.0
    w = w - np.array([0, w[:, 1].mean(), 0])
    tid, _, _ = render_view(cam, w, f)
    hit = tid >= 0
    n = np.cross(w[f[:, 1]] - w[f[:, 0]], w[f[:, 2]] - w[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    rgba = np.zeros((260, 200, 4), np.uint8)
    rgba[hit, :3] = np.clip((0.35 + 0.65 * np.abs(n @ cam.to_camera))[tid[hit]][:, None] * np.array([200, 120, 90.0]), 0, 255).astype(np.uint8)
    rgba[hit, 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "pic.png")
    cap = {}
    orig = PT.project_views_into_atlas

    def spy(tex, mesh, views, **kw):
        out, rep = orig(tex, mesh, views, **kw)
        cap.update(tin=tex.copy(), tout=out, mesh=mesh, rep=rep)
        return out, rep

    monkeypatch.setattr(PT, "project_views_into_atlas", spy)
    build_asset(str(tmp_path / "pic.png"), str(tmp_path / "pack"),
                AssetOptions(kind="character", name="fig", proxy=(v, f), proxy_yaw=0.0, elev=20.0, lods=2, preview=False, atlas=512,
                             extra_views=[_coloured_view(180, (40, 200, 60)), _coloured_view(90, (230, 220, 40))]), log=lambda s: None)
    tin, tout = cap["tin"], cap["tout"]
    cov, _, N = PT.texel_surface(*[np.asarray(a) for a in (cap["mesh"][0], cap["mesh"][1], cap["mesh"][2])], tin.shape[:2])
    mean = lambda sel, t: t[sel].reshape(-1, 3).mean(0)
    front = cov & (N @ np.array([0, 0, 1.0]) > 0.7)
    back = cov & (N @ np.array([0, 0, -1.0]) > 0.7)
    left = cov & (N @ np.array([1.0, 0, 0]) > 0.7)
    assert np.abs(mean(front, tout) - mean(front, tin)).max() < 3                      # the picture itself is untouched
    assert mean(back, tout)[1] > mean(back, tin)[1] + 25 and mean(back, tout)[0] < mean(back, tin)[0] - 25     # greener
    assert mean(left, tout)[2] < mean(left, tin)[2] - 5 or mean(left, tout)[1] > mean(left, tin)[1] + 10       # towards yellow
    assert all(vw["registration_iou"] > 0.8 for vw in cap["rep"]["views"])


def test_a_view_that_does_not_belong_to_the_mesh_is_skipped_not_painted_on(tmp_path):
    v, f = TRUTH
    from aura_white.asset.meshio import read_gltf  # noqa: F401
    bar = np.zeros((256, 256), bool)
    bar[10:240, 120:130] = True
    bad = PT.ExtraView("back", 180.0, 0.0, np.full((256, 256, 3), 255, np.uint8), bar, 1.0, "user")
    uv = np.random.default_rng(0).random((len(v), 2))
    atlas = np.full((64, 64, 3), 100, np.uint8)
    out, rep = PT.project_views_into_atlas(atlas, (v, f, uv), [bad])
    assert "skipped" in rep["views"][0] and np.array_equal(out[: 1, : 1], atlas[: 1, : 1])


# ------------------------------------------------------------------------------------------ asset hook
def test_asset_with_generate_auto_builds_depth_where_the_old_default_built_a_slab(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    photo(0).save(tmp_path / "knight.png")
    depth = lambda d: audit_mesh(*[np.asarray(a) for a in read_gltf(str(next(d.glob("*.glb"))))[:2]], expect_solid=False)
    build_asset(str(tmp_path / "knight.png"), str(tmp_path / "old"), AssetOptions(kind="character", name="k", lods=2, preview=False, atlas=512), log=lambda s: None)
    build_asset(str(tmp_path / "knight.png"), str(tmp_path / "new"),
                AssetOptions(kind="character", name="k", lods=2, preview=False, atlas=512, generate="auto",
                             generate_opts=dict(quality="standard", candidates=2)), log=lambda s: None)
    old, new = depth(tmp_path / "old"), depth(tmp_path / "new")
    assert old.profile in ("mirrored-relief", "extruded", "thin") or old.depth_ratio < 0.2
    assert new.depth_ratio > 2 * old.depth_ratio and new.profile == "solid"


def test_asset_generate_auto_refuses_to_flatten_when_no_generator_exists(tmp_path):
    photo(0).save(tmp_path / "k.png")
    with pytest.raises(S.NoBackend):
        build_asset(str(tmp_path / "k.png"), str(tmp_path / "o"), AssetOptions(kind="character", name="k", lods=2, preview=False, generate="auto"),
                    log=lambda s: None)


def test_analytic_kinds_are_not_sent_to_the_generator(tmp_path):
    img = np.zeros((128, 128, 4), np.uint8)
    yy, xx = np.mgrid[:128, :128]
    disc = (xx - 64) ** 2 + (yy - 64) ** 2 < 50 ** 2
    img[disc] = (80, 120, 200, 255)
    Image.fromarray(img, "RGBA").save(tmp_path / "ball.png")
    m = build_asset(str(tmp_path / "ball.png"), str(tmp_path / "o"), AssetOptions(kind="sphere", name="ball", lods=2, preview=False, atlas=256,
                                                                                generate="auto"), log=lambda s: None)
    assert m["kind"] == "sphere"                                                        # a true sphere stays an analytic sphere


# ------------------------------------------------------------------------------------------ CLI
def test_cli_parses_the_new_commands():
    from aura_white import cli
    a = cli.build_parser().parse_args(["make", "a.png", "b.png", "--quality", "ultra", "--tier", "max", "--allow-basic", "--seed", "3"])
    assert a.fn is cli.cmd_make and a.quality == "ultra" and a.tier == "max" and a.allow_basic and a.seed == 3 and a.inputs == ["a.png", "b.png"]
    r = cli.build_parser().parse_args(["refine", "out/knight", "--add", "back=b.png"])
    assert r.fn is cli.cmd_refine and r.add == ["back=b.png"]
    assert cli.build_parser().parse_args(["studio", "x.png"]).fidelity == "standard"        # the slab is no longer the default
    assert cli.build_parser().parse_args(["asset", "x.png"]).allow_flat is False


def test_a_bare_picture_path_goes_to_make(tmp_path, monkeypatch):
    from aura_white import cli
    photo(0).save(tmp_path / "k.png")
    seen = {}
    monkeypatch.setattr(cli, "cmd_make", lambda a: seen.setdefault("inputs", a.inputs) and 0)
    assert cli.main([str(tmp_path / "k.png")]) == 0 or seen
    assert seen["inputs"] == [str(tmp_path / "k.png")]
