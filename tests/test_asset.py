"""Game-asset pipeline (aura_white.asset): camera, rasteriser, mesh loading, mesh-lock, glTF writer, CLI.  Small synthetic
data only - no model weights, no network."""
import json
import struct

import numpy as np
import pytest
import trimesh
from PIL import Image

from aura_white.asset import collision, meshio
from aura_white.asset.camera import OrthoCamera, rasterize_ortho, render_view
from aura_white.asset.gltf import write_glb
from aura_white.asset.meshfill import harmonic_colours
from aura_white.asset.meshlock import fit_pose, front_faces, front_layout, lock_outline
from aura_white.asset.pipeline import AssetOptions, build_asset


# ---------------------------------------------------------------------------------------------------- helpers
def _figure():
    """An asymmetric standing figure: ellipsoid body, a box arm on one side, a head."""
    body = trimesh.creation.icosphere(2)
    body.apply_scale([0.45, 0.8, 0.3])
    arm = trimesh.creation.box((0.7, 0.12, 0.12))
    arm.apply_translation([0.55, 0.25, 0.0])
    head = trimesh.creation.icosphere(1)
    head.apply_scale(0.22)
    head.apply_translation([0.0, 1.0, 0.0])
    m = trimesh.util.concatenate([body, arm, head])
    v = np.asarray(m.vertices, np.float64)
    v[:, 1] -= v[:, 1].min()
    return v, np.asarray(m.faces, np.int64)


def _picture(tmp_path, v, f, elev=20.0, scale=180.0, size=(260, 320)):
    """A 'drawing' of the figure: its flat-shaded orthographic render on a transparent background."""
    cam = OrthoCamera(size[0], size[1], elev)
    w = v * scale
    w = w - np.array([0, w[:, 1].mean(), 0])
    tid, _, _ = render_view(cam, w, f)
    hit = tid >= 0
    n = np.cross(w[f[:, 1]] - w[f[:, 0]], w[f[:, 2]] - w[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    shade = 0.35 + 0.65 * np.abs(n @ cam.to_camera)
    rgb = np.zeros((size[1], size[0], 4), np.uint8)
    base = np.array([200, 120, 90], np.float64)
    rgb[hit, :3] = np.clip(shade[tid[hit]][:, None] * base, 0, 255).astype(np.uint8)
    rgb[hit, 3] = 255
    p = tmp_path / "figure.png"
    Image.fromarray(rgb, "RGBA").save(p)
    return p, hit


def _glb_json(path):
    d = open(path, "rb").read()
    assert d[:4] == b"glTF"
    ln, ct = struct.unpack("<I4s", d[12:20])
    assert ct == b"JSON"
    return json.loads(d[20:20 + ln])


# ---------------------------------------------------------------------------------------------------- camera / raster
def test_camera_roundtrip_and_ground_plane():
    cam = OrthoCamera(200, 100, 30.0)
    P = np.array([[10.0, 5.0, -3.0], [-40.0, 12.0, 8.0], [0.0, 0.0, 0.0]])
    x, y, t = cam.project(P)
    assert np.allclose(cam.unproject(x, y, t), P, atol=1e-9)
    G = cam.ground_points(np.array([100.0, 60.0]), np.array([80.0, 90.0]), Y=0.0)
    assert np.allclose(G[:, 1], 0.0, atol=1e-9)
    x2, y2, _ = cam.project(G)
    assert np.allclose(x2, [100.0, 60.0]) and np.allclose(y2, [80.0, 90.0])


def test_rasterizer_depth_order_and_coverage():
    xy = np.array([[2.0, 2.0], [18.0, 2.0], [2.0, 18.0], [8.0, 8.0], [18.0, 8.0], [8.0, 18.0]])
    depth = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    tid, bary, z = rasterize_ortho(xy, depth, np.array([[0, 1, 2], [3, 4, 5]]), 20, 20)
    assert tid[3, 3] == 0 and tid[12, 10] == 1             # the small triangle (nearer = smaller depth) wins where both cover
    assert np.isnan(z[19, 19]) and tid[19, 19] == -1
    assert np.allclose(bary[tid >= 0].sum(1), 1.0, atol=1e-4)
    assert abs(float(z[12, 10]) - 1.0) < 1e-4


# ---------------------------------------------------------------------------------------------------- mesh io
def test_weld_compact_and_specks():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1e-9]], float)
    f = np.array([[0, 1, 2], [1, 3, 2], [4, 1, 2]])
    v2, f2 = meshio.weld(v, f, 1e-6)
    assert len(v2) == 4 and len(f2) == 3                      # vertex 4 merged into vertex 0; no face degenerates
    big = trimesh.creation.icosphere(2)
    tiny = trimesh.creation.icosphere(0)
    tiny.apply_scale(0.01)
    tiny.apply_translation([3, 0, 0])
    m = trimesh.util.concatenate([big, tiny])
    vv, ff = meshio.drop_specks(np.asarray(m.vertices), np.asarray(m.faces), min_share=0.05, min_faces=30)
    assert len(ff) == len(big.faces)


def test_simplify_reduces_and_survives_without_quadric(monkeypatch):
    m = trimesh.creation.icosphere(4)
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    v2, f2 = meshio.simplify(v, f, 800)
    assert 300 < len(f2) < 2500 and len(f2) < len(f)
    monkeypatch.setitem(__import__("sys").modules, "fast_simplification", None)    # ImportError on import
    v3, f3 = meshio.simplify(v, f, 800)
    assert 100 < len(f3) < len(f)


def test_gltf_roundtrip_with_texture_colours(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    f = np.array([[0, 1, 2], [1, 3, 2]])
    uv = np.array([[0.1, 0.9], [0.9, 0.9], [0.1, 0.1], [0.9, 0.1]], np.float32)
    tex = np.zeros((16, 16, 3), np.uint8)
    tex[...] = (200, 40, 40)
    p = tmp_path / "quad.glb"
    write_glb(p, [dict(name="q", verts=v, faces=f, uv=uv, material=0)], [dict(name="m", base="a")], {"a": tex})
    v2, f2 = meshio.load_mesh(p)
    assert v2.shape == (4, 3) and f2.shape == (2, 3)
    v3, f3, cs = meshio.load_mesh_coloured(p, n_samples=200)
    assert cs is not None and len(cs[0]) >= 4
    assert np.abs(cs[1].astype(int) - np.array([200, 40, 40])).max() <= 2


def test_obj_loader_triangulates_quads(tmp_path):
    p = tmp_path / "q.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    v, f = meshio.load_mesh(p)
    assert len(v) == 4 and len(f) == 2


def test_glb_writer_lods_and_extensions(tmp_path):
    m = trimesh.creation.icosphere(1)
    v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces)
    p = tmp_path / "x.glb"
    n = write_glb(p, [dict(name="a_LOD0", verts=v, faces=f), dict(name="a_LOD1", verts=v, faces=f[:10]),
                      dict(name="a_Collider_01", verts=v, faces=f)], [], {}, lod_group=[0, 1])
    j = _glb_json(p)
    assert n == p.stat().st_size and len(j["nodes"]) == 3
    assert j["nodes"][0]["extensions"]["MSFT_lod"]["ids"] == [1]
    assert 1 not in j["scenes"][0]["nodes"]                     # the lower LOD hangs off LOD0, not the scene


# ---------------------------------------------------------------------------------------------------- mesh-lock
def test_harmonic_colours_interpolate_along_the_surface():
    n = 11
    v = np.stack([np.linspace(0, 10, n), np.zeros(n), np.zeros(n)], 1)
    top = np.stack([np.linspace(0, 10, n), np.ones(n), np.zeros(n)], 1)
    V = np.concatenate([v, top])
    F = []
    for i in range(n - 1):
        F += [[i, i + 1, n + i], [i + 1, n + i + 1, n + i]]
    fixed = np.zeros(2 * n, bool)
    fixed[[0, n]] = fixed[[n - 1, 2 * n - 1]] = True
    col = np.zeros((2 * n, 3))
    col[[0, n]] = (0, 0, 0)
    col[[n - 1, 2 * n - 1]] = (200, 100, 50)
    out = harmonic_colours(V, np.array(F), fixed, col)
    mid = out[5]
    assert np.allclose(out[0], 0) and np.allclose(out[n - 1], (200, 100, 50))
    assert np.all(np.diff(out[:n, 0]) >= -1e-6) and 60 < mid[0] < 140


def test_fit_pose_recovers_a_turned_mesh(tmp_path):
    v, f = _figure()
    _, hit = _picture(tmp_path, v, f, elev=20.0)
    th = np.radians(90.0)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    turned = v @ R.T * 3.7 + np.array([5.0, 2.0, 1.0])        # a different frame, scale and orientation
    pose = fit_pose(hit.astype(np.float32), turned, f, elev=20.0)
    assert pose.iou > 0.9
    assert abs(pose.elev - 20.0) < 1e-6


def test_lock_outline_keeps_the_mesh_inside_the_matte():
    cam = OrthoCamera(100, 100, 0.0)
    alpha = np.zeros((100, 100), np.float32)
    alpha[30:70, 30:70] = 1.0
    v = np.array([[-10, 0, 0], [10, 0, 0], [0, 10, 0], [18, 0, 0], [0, -18, 0], [14, 14, 0], [60, 60, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2], [0, 4, 1], [1, 5, 2], [3, 6, 5]])
    v2, f2, info = lock_outline(v, f, cam, alpha, margin=3.0)
    assert info["keep"].dtype == bool and info["keep"].sum() == len(f2)
    px, py, _ = cam.project(v2)
    inside = (px >= 29.4) & (px <= 70.6) & (py >= 29.4) & (py <= 70.6)
    assert inside[np.unique(f2)].all()
    assert len(f2) < len(f)                                   # the faces reaching far outside were dropped


def test_front_layout_keeps_the_picture_at_one_to_one_and_shrinks_the_sides_first():
    sf, sm = front_layout(833, 1507, 2048, 2048, 2048)
    assert sf == 1.0 and 0.4 <= sm < 1.0
    sf2, sm2 = front_layout(1500, 1500, 2048, 2048, 2048)              # cannot share the atlas at 1:1 -> the picture shrinks a little
    assert 0.5 <= sf2 < 1.0 and sm2 > 0.2
    sf3, sm3 = front_layout(5000, 5000, 2048, 2048, 2048)               # absurd crop: still returns something that fits
    assert sf3 <= (2048 - 16) / 5000 + 1e-9 and sm3 <= 0.3


def test_front_faces_are_the_ones_the_camera_sees_head_on():
    cam = OrthoCamera(100, 100, 0.0)
    # two quads (4 triangles): one facing the camera at z=1, one edge-on, one hidden behind the first
    v = np.array([[-20, -20, 1], [20, -20, 1], [20, 20, 1], [-20, 20, 1],           # front quad
                  [30, -20, 0], [30, -20, -10], [30, 20, -10], [30, 20, 0],          # a wall parallel to the view direction
                  [-10, -10, -5], [10, -10, -5], [10, 10, -5], [-10, 10, -5]], float)   # hidden behind the front quad
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11]])
    flags = front_faces(v, f, cam)
    assert flags[:2].all() and not flags[2:4].any() and not flags[4:].any()


# ---------------------------------------------------------------------------------------------------- whole pipeline
def test_collision_hulls_are_convex_and_closed():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(400, 3))
    for g in collision.kmeans_hulls(pts, 4):
        hv, hf, vol = collision.hull(g)
        assert vol > 0
        e = {}
        for tri in hf:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                e[(tri[a], tri[b])] = e.get((tri[a], tri[b]), 0) + 1
        assert all(e.get((b, a), 0) == 1 for (a, b) in e)     # every edge is shared by exactly two oppositely wound faces


def test_mesh_lock_pipeline_end_to_end(tmp_path):
    v, f = _figure()
    img, hit = _picture(tmp_path, v, f, elev=20.0)
    out = tmp_path / "pack"
    logs = []
    m = build_asset(img, out, AssetOptions(kind="character", name="fig", proxy=(v, f), proxy_yaw=0.0, elev=20.0, lods=3,
                                           max_faces=2500, atlas=256, preview=False, height=1.7), log=logs.append)
    assert m["tier"] == "M" and m["kind"] == "character"
    for rel in ("fig.glb", "fig_unlit.glb", "game.json", "report.md", "textures/fig_albedo.png",
                "textures/fig_normal.png", "textures/fig_orm.png", "obj/fig.obj"):
        assert (out / rel).exists(), rel
    assert [n["name"] for n in _glb_json(out / "fig.glb")["nodes"]] == ["fig"]
    lods = _glb_json(out / "lods" / "fig_lods.glb")
    assert [n["name"] for n in lods["nodes"]][:2] == ["fig_LOD0", "fig_LOD1"]
    assert "MSFT_lod" in lods["nodes"][0]["extensions"]
    assert any((out / "collision").iterdir())
    assert m["fidelity_reference_view"]["iou"] > 0.9
    assert abs(m["size_m"][1] - 1.7) < 0.02
    bad = [c for c in m["checks"] if c["ok"] is False]
    assert not bad, bad
    assert json.loads((out / "game.json").read_text())["name"] == "fig"
    lod = [x["triangles"] for x in m["lods"]]
    assert lod == sorted(lod, reverse=True) and len(lod) == 3


def test_mesh_lock_uses_a_back_picture_and_the_meshs_own_colours(tmp_path):
    from aura_white.asset.meshlock import fit_pose  # noqa: F401  (import check only)
    v, f = _figure()
    img, hit = _picture(tmp_path, v, f, elev=20.0)
    # a textured proxy: one flat colour per vertex height band, so the sides/back have a colour the picture does not have
    tex = np.zeros((8, 8, 3), np.uint8)
    tex[...] = (30, 160, 60)
    uv = np.full((len(v), 2), 0.5, np.float32)
    proxy = tmp_path / "proxy.glb"
    write_glb(proxy, [dict(name="p", verts=v.astype(np.float32), faces=f, uv=uv, material=0)],
              [dict(name="m", base="a")], {"a": tex})
    out = tmp_path / "pack_tex"
    m = build_asset(img, out, AssetOptions(kind="character", name="fig", proxy=str(proxy), proxy_yaw=0.0, elev=20.0,
                                           lods=2, max_faces=2500, atlas=256, preview=False), log=lambda s: None)
    assert m["info"].get("source_colours", 0) > 0
    alb = np.asarray(Image.open(out / "textures" / "fig_albedo.png").convert("RGB")).reshape(-1, 3).astype(int)
    green = (np.abs(alb - np.array([30, 160, 60])).max(1) < 25).mean()
    assert green > 0.05                                      # the hidden half of the atlas carries the mesh's own colour
    out2 = tmp_path / "pack_back"
    back = tmp_path / "back.png"
    Image.open(img).transpose(Image.FLIP_LEFT_RIGHT).save(back)
    m2 = build_asset(img, out2, AssetOptions(kind="character", name="fig", proxy=(v, f), proxy_yaw=0.0, elev=20.0, lods=2,
                                             max_faces=2500, atlas=256, preview=False, back_image=str(back)),
                     log=lambda s: None)
    assert not [c for c in m2["checks"] if c["ok"] is False]


def test_analytic_sphere_pack_and_cli(tmp_path):
    from aura_white.cli import main
    n = 160
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5)
    img = np.zeros((n, n, 4), np.uint8)
    inside = r < 64
    img[inside, 0] = np.clip(120 + 80 * np.sin(xx[inside] / 7.0), 0, 255)
    img[inside, 1] = 30
    img[inside, 2] = 40
    img[inside, 3] = 255
    p = tmp_path / "ball.png"
    Image.fromarray(img, "RGBA").save(p)
    out = tmp_path / "ballpack"
    rc = main(["asset", str(p), "-o", str(out), "--lods", "2", "--atlas", "256", "--no-preview", "--name", "ball", "--kind", "sphere"])
    assert rc == 0
    man = json.loads((out / "game.json").read_text())
    assert man["tier"] == "A" and man["kind"] == "sphere"
    assert man["fidelity_reference_view"]["iou"] > 0.98
    assert (out / "ball.glb").exists()
