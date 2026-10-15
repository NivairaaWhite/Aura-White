import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("scipy")

from aura_white.forge import complete, morph, patch, snap, softraster as sr, wires


def ring_mask(size=200, r_out=60, r_in=20, cx=100, cy=100):
    yy, xx = np.mgrid[:size, :size]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (d2 < r_out ** 2) & (d2 > r_in ** 2)


def test_contours_area_sign_and_hole():
    f = ring_mask().astype(np.float32)
    loops = morph.contours(f, 0.5)
    assert len(loops) == 2
    areas = [morph.polygon_area(l) for l in loops]
    assert (areas[0] > 0) != (areas[1] > 0), "outer boundary and inner hole must have opposite signed area"


def test_simplify_closed_reduces_points_and_stays_closed():
    theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    xy = np.stack([100 + 60 * np.cos(theta), 100 + 60 * np.sin(theta)], 1)
    out = morph.simplify_closed(xy, tol=0.5, max_len=10)
    assert 3 < len(out) < len(xy)
    seg = np.hypot(*np.diff(np.vstack([out, out[:1]]), axis=0).T)
    assert seg.max() < 15


def test_thin_produces_1px_skeleton_with_correct_topology():
    mask = np.zeros((60, 200), bool)
    mask[28:32, 10:190] = True                      # a straight bar
    sk = morph.thin(mask)
    assert sk.sum() > 0
    row_counts = sk.sum(0)
    assert (row_counts[20:180] <= 1).all(), "skeleton of a straight bar must be 1 pixel wide"


def test_relief_plate_is_closed_and_face_normals_face_out():
    mask = ring_mask(size=120, r_out=45, r_in=15, cx=60, cy=60)
    from scipy import ndimage as ndi

    alpha = ndi.gaussian_filter(mask.astype(np.float32), 0.7)
    dt = morph.edt(mask)
    half = patch.plate_profile(dt, 6.0, 4.0)
    m = patch.build_relief(alpha, half)
    assert m is not None and len(m.faces) > 0
    import trimesh

    tm = trimesh.Trimesh(m.verts, m.faces, process=False)
    assert tm.is_watertight and tm.is_winding_consistent
    assert tm.volume > 0


def test_chain_link_is_a_closed_ring_of_faces():
    v, f = wires.link_mesh(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), half_len=2.0, end_radius=1.0,
                           wire_radius=0.3)
    import trimesh

    tm = trimesh.Trimesh(v, f, process=False)
    assert tm.is_watertight and tm.is_winding_consistent


def test_chain_mesh_places_the_requested_number_of_links():
    P = np.stack([np.linspace(0, 20, 50), np.zeros(50), np.zeros(50)], 1)
    v, f, n = wires.chain_mesh(P, width=2.0, view_dir=np.array([0.0, 0.0, -1.0]), pitch_ratio=1.3)
    assert n == int(20 / (1.3 * 2.0))
    assert len(v) > 0 and len(f) > 0


def test_tube_mesh_follows_a_curved_polyline():
    t = np.linspace(0, np.pi, 40)
    P = np.stack([t, np.sin(t), np.zeros_like(t)], 1) * 5
    v, f = wires.tube_mesh(P, radius=0.3)
    assert len(v) > 0
    assert np.linalg.norm(v.mean(0) - P.mean(0)) < 3.0


def scythe_like_mask(size=256):
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    d.pieslice((40, 40, 220, 220), 200, 340, fill=255, width=0)
    d.line((130, 130, 130, 250), fill=255, width=14)
    d.line((90, 150, 90, 240), fill=255, width=4)                # a thin cord
    return np.asarray(im) > 0


def test_find_wires_detects_the_thin_cord_not_the_bulk():
    mask = scythe_like_mask()
    rgb = np.full(mask.shape + (3,), 255, np.uint8)
    rgb[mask] = (80, 80, 90)
    found, used = wires.find_wires(mask, rgb, bg=np.array([255.0, 255.0, 255.0]))
    assert len(found) >= 1
    assert any(w.info["fat_width"] < 10 for w in found)
    assert used.sum() < mask.sum()


def _square_cam():
    class Cam:
        w2c = np.eye(4)
        w2c = w2c.copy()
        f = 1.0 / np.tan(np.radians(15))
        cx = cy = 0.0

    c = Cam()
    c.w2c[2, 3] = -3.0
    return c


def test_softraster_projects_and_unprojects_round_trip():
    cam = _square_cam()
    pts = np.array([[0.1, 0.2, 0.0], [-0.3, 0.1, 0.05]])
    xy, w = sr.project(cam, pts, 256, 256)
    back = sr.unproject(cam, xy[:, 0], xy[:, 1], w, 256, 256)
    assert np.allclose(back, pts, atol=1e-6)


def test_rasterize_covers_a_triangle_and_z_sorts():
    xy = np.array([[10.0, 10.0], [90.0, 10.0], [50.0, 90.0], [10.0, 10.0], [90.0, 10.0], [50.0, 90.0]])
    w = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    tri, bary, depth = sr.rasterize(xy, w, tris, 100, 100)
    assert (tri[50, 50] == 1), "the nearer triangle (w=1) must win the z-test"
    assert np.isfinite(depth[50, 50]) and depth[50, 50] < 2.0


def test_residual_completion_adds_the_missing_chain(tmp_path=None):
    size = 256
    mask = scythe_like_mask(size)
    rgb = np.full((size, size, 3), 255, np.uint8)
    rgb[mask] = (90, 90, 100)
    alpha = mask.astype(np.float32)
    bulk = morph.opening(mask, 8.0)                  # a "neural mesh" that lost the thin cord
    mesh_mask = bulk
    depth = np.where(mesh_mask, 3.0, np.inf).astype(np.float32)
    cam = _square_cam()
    res = complete.complete_residual(rgb, alpha, mesh_mask, depth, cam, complete.CompletionOptions())
    assert res.n_faces > 0
    assert any(p["kind"] in ("tube", "chain") for p in res.parts)


def test_snap_pulls_sphere_silhouette_onto_an_ellipse():
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=4, radius=0.6)
    cam = _square_cam()
    size = 300
    yy, xx = np.mgrid[:size, :size]
    half = 3.0 * np.tan(np.radians(15))
    X = ((xx + 0.5) / size * 2 - 1) * half
    Y = (1 - (yy + 0.5) / size * 2) * half
    ref = (((X / 0.75) ** 2 + (Y / 0.55) ** 2) <= 1).astype(np.float32)
    res = snap.snap_to_silhouette(np.asarray(m.vertices), np.asarray(m.faces), cam, ref, max_shift_px=60)
    assert res.iou_after > res.iou_before + 0.1
