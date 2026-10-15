import numpy as np

from aura_white.fidelity import build_reference_locked_mesh
from aura_white.raster.camera import orbit_camera


def _geometrically_closed(r) -> bool:
    """Every edge occurs exactly twice once vertices that share a position (texture seams) are welded."""
    _, weld = np.unique(np.round(r.verts.astype(np.float64), 6), axis=0, return_inverse=True)
    f = weld[r.faces]
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def test_source_locked_mesh_is_closed_and_uv_bound():
    v = np.array([[0, -0.8, -0.5], [0, 0.8, -0.5], [0, 0.8, 0.5], [0, -0.8, 0.5]], np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    alpha = np.zeros((128, 128), np.float32)
    alpha[24:104, 40:88] = 1
    cam = orbit_camera(0, 0, 4, 30)
    r = build_reference_locked_mesh(alpha, v, f, cam, working_res=128, point_step=4, boundary_step=1)
    assert len(r.verts) > 0 and len(r.faces) > 0
    assert np.isfinite(r.verts).all()
    assert np.all((r.uv >= 0) & (r.uv <= 1))
    assert _geometrically_closed(r)
    assert len(r.uv) == len(r.verts)


def _flat_source():
    v = np.array([[0, -20, -20], [0, 20, -20], [0, 20, 20], [0, -20, 20]], np.float64)
    return v, np.array([[0, 1, 2], [0, 2, 3]], np.int64)


def _hard_mask():
    """Thin diagonal chain, a 1-px wire, a detached island, an enclosed hole and a pinch point: the shapes that
    used to be lost to down-sampling and 'largest contour only' extraction."""
    m = np.zeros((160, 120), bool)
    m[20:110, 30:80] = True
    m[50:70, 45:60] = False                                   # enclosed hole
    for i in range(45):                                       # 2-px wide diagonal chain
        m[min(112 + i // 2, 159), 20 + i:22 + i] = True
    m[10:130, 100] = True                                     # 1-px wire
    m[130:138, 60:68] = True                                  # detached island
    m[110, 79:96] = True                                      # bridge that pinches against the body corner
    return m


def _render_iou(r, cam, mask):
    from aura_white.forge.softraster import render_camera
    m, _, _ = render_camera(cam, r.verts.astype(np.float64), r.faces.astype(np.int64), *mask.shape)
    return float((m & mask).sum() / (m | mask).sum())


def test_source_locked_reproduces_thin_islands_holes_and_pinches():
    mask = _hard_mask()
    v, f = _flat_source()
    cam = orbit_camera(0, 0, 4, 30)
    r = build_reference_locked_mesh(mask.astype(np.float32), v, f, cam, working_res=256, point_step=8, boundary_step=1)
    assert r.report["closed"], r.report
    assert r.report["open_edges"] == 0 and r.report["shared_edges"] == 0
    assert _render_iou(r, cam, mask) >= 0.999


def test_source_locked_never_downsamples_below_the_image():
    mask = np.zeros((300, 200), bool)
    mask[40:260, 60:140] = True
    mask[100:103, 140:190] = True                             # 3-px arm that a 128-px working copy would erase
    v, f = _flat_source()
    r = build_reference_locked_mesh(mask.astype(np.float32), v, f, orbit_camera(0, 0, 4, 30), working_res=2048)
    assert tuple(r.report["working_resolution"]) == (200, 300)
    assert r.report["mask_coverage_iou"] >= 0.999


def test_source_texture_is_native_and_has_no_background_halo():
    from aura_white.fidelity import source_texture
    rgb = np.ones((40, 30, 3), np.float32)                    # white background
    rgb[10:30, 8:22] = (0.8, 0.1, 0.1)                        # red object
    alpha = np.zeros((40, 30), np.float32)
    alpha[10:30, 8:22] = 1.0
    alpha[10, 8:22] = 0.5                                     # anti-aliased top edge, contaminated with white
    rgb[10, 8:22] = (0.9, 0.55, 0.55)
    tex = source_texture(rgb, alpha, solid=0.98)              # the opt-in "clean rim" setting
    assert tex.shape == (40, 30, 3)                           # native size, no resampling
    assert np.array_equal(tex[15:28, 10:20], np.round(rgb[15:28, 10:20] * 255).astype(np.uint8))   # solid pixels untouched
    outside = tex[alpha < 0.98].astype(np.float32) / 255
    assert (outside[:, 1] < 0.2).all()                        # halo and background are red, not white
    exact = source_texture(rgb, alpha)                        # default: every pixel the mesh can show is the original
    assert np.array_equal(exact[alpha >= 0.5], np.round(rgb[alpha >= 0.5] * 255).astype(np.uint8))
    assert (exact[alpha < 0.5][:, 1] < 200).all()             # outside is filled from kept pixels, never the white background


def test_side_walls_sample_inside_the_artwork_not_the_rim():
    """Caps sample the exact silhouette (identical reference view); walls sample a few pixels inside."""
    v, f = _flat_source()
    alpha = np.zeros((128, 128), np.float32)
    alpha[24:104, 40:88] = 1
    r = build_reference_locked_mesh(alpha, v, f, orbit_camera(0, 0, 4, 30), working_res=128, point_step=4, boundary_step=1)
    assert r.report["wall_uv_inward"] and _geometrically_closed(r)
    u = r.uv * np.array([128.0, 128.0])
    z = r.verts[:, 0]                                         # depth axis of the flat source (x)
    wall = np.zeros(len(r.verts), bool)
    wall[np.unique(r.faces[-1000:])] = False
    rim = (np.abs(u[:, 0] - 40) < 0.6) | (np.abs(u[:, 0] - 88) < 0.6) | (np.abs(u[:, 1] - 24) < 0.6) | (np.abs(u[:, 1] - 104) < 0.6)
    assert rim.any()                                          # cap vertices still sit exactly on the outline
    assert (u[:, 0] >= 39.4).all() and (u[:, 0] <= 88.6).all()
    assert np.isfinite(z).all()


def test_defringe_removes_background_blend_but_keeps_white_parts():
    from aura_white.fidelity import source_texture
    rgb = np.zeros((40, 40, 3), np.float32)
    rgb[:, :] = (0.7, 0.1, 0.1)                               # red object
    rgb[:, :3] = (0.97, 0.97, 0.97)                           # 1..3 px halo of old white background at the left rim
    rgb[10:20, 20:30] = 1.0                                   # a genuinely white part deep inside
    alpha = np.zeros((40, 40), np.float32)
    alpha[:, 2:38] = 1.0
    out = source_texture(rgb, alpha, max_side=64, defringe=True)
    assert out.dtype == np.uint8
    assert out[20, 2, 0] < 230 and out[20, 2, 1] < 100        # rim pixel now carries the object colour (0-255 scale)
    assert (out[15, 25] == 255).all()                         # white part untouched
    kept = source_texture(rgb, alpha, max_side=64)             # default is pixel-exact
    assert kept[20, 2, 1] > 200                               # the pale rim is preserved unless asked otherwise


def test_pinched_mask_stays_manifold_after_welding_by_position():
    """Two parts touching at one pixel corner: exporters/DCC tools weld by position, which must not create 4-face edges."""
    v, f = _flat_source()
    m = np.zeros((64, 64), bool)
    m[8:30, 8:30] = True
    m[30:52, 30:52] = True                                     # diagonal contact at a single corner
    r = build_reference_locked_mesh(m.astype(np.float32), v, f, orbit_camera(0, 0, 4, 30), working_res=64,
                                    point_step=4, boundary_step=1)
    assert r.report["pinch_vertices_split"] > 0
    assert _geometrically_closed(r)
