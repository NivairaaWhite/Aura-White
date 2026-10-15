import numpy as np
import trimesh

from aura_white.forge import snap, softraster as sr
from aura_white.raster.camera import orbit_camera


def _sphere():
    m = trimesh.creation.icosphere(subdivisions=4, radius=0.6)
    return np.asarray(m.vertices), np.asarray(m.faces)


def _ellipse_mask(size, rx, ry):
    yy, xx = np.mgrid[:size, :size]
    half = 3.0 * np.tan(np.radians(15))
    X = ((xx + 0.5) / size * 2 - 1) * half
    Y = (1 - (yy + 0.5) / size * 2) * half
    return (((X / rx) ** 2 + (Y / ry) ** 2) <= 1).astype(np.float32)


def test_snap_to_silhouette_matches_single_view_wrapper():
    """snap_to_silhouette must behave exactly like snap_to_silhouettes with one view (no drift between them)."""
    v, f = _sphere()
    cam = orbit_camera(0, 0, 3.0, 30)
    ref = _ellipse_mask(300, 0.75, 0.55)
    a = snap.snap_to_silhouette(v, f, cam, ref, max_shift_px=60, iterations=2)
    b = snap.snap_to_silhouettes(v, f, [(cam, ref)], max_shift_px=60, iterations=2)
    assert np.allclose(a.verts, b.verts)
    assert a.iou_before == pytest_approx(b.iou_before)
    assert a.iou_after == pytest_approx(b.iou_after)


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-9)


def test_joint_snap_improves_a_view_the_single_view_never_saw():
    v, f = _sphere()
    size = 300
    cam_front, ref_front = orbit_camera(0, 0, 3.0, 30), _ellipse_mask(size, 0.75, 0.55)
    cam_side, ref_side = orbit_camera(90, 0, 3.0, 30), _ellipse_mask(size, 0.45, 0.55)

    single = snap.snap_to_silhouette(v, f, cam_front, ref_front, max_shift_px=60, iterations=3)
    m_side_single, _, _ = sr.render_camera(cam_side, single.verts, f, size)
    iou_side_single = float((m_side_single & (ref_side >= 0.5)).sum() / (m_side_single | (ref_side >= 0.5)).sum())

    joint = snap.snap_to_silhouettes(v, f, [(cam_front, ref_front), (cam_side, ref_side)], max_shift_px=60,
                                     iterations=3)
    m_side_joint, _, _ = sr.render_camera(cam_side, joint.verts, f, size)
    iou_side_joint = float((m_side_joint & (ref_side >= 0.5)).sum() / (m_side_joint | (ref_side >= 0.5)).sum())

    assert iou_side_joint > iou_side_single + 0.1, "the side view should improve a lot once it is actually used"

    m_front_joint, _, _ = sr.render_camera(cam_front, joint.verts, f, size)
    iou_front_joint = float((m_front_joint & (ref_front >= 0.5)).sum() / (m_front_joint | (ref_front >= 0.5)).sum())
    assert iou_front_joint > 0.99, "the front view should not regress just because a second view was added"


def test_joint_snap_handles_a_vertex_visible_in_no_view_gracefully():
    v, f = _sphere()
    size = 200
    cam = orbit_camera(0, 0, 3.0, 30)
    empty_ref = np.zeros((size, size), np.float32)
    res = snap.snap_to_silhouettes(v, f, [(cam, empty_ref)], max_shift_px=30, iterations=2)
    assert np.isfinite(res.verts).all()
