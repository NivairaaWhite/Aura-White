import numpy as np
import trimesh

from aura_white import evaluation as E


def test_identical_mesh_scores_perfectly():
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    r = E.mesh_metrics(m.vertices, m.faces, m.vertices, m.faces, tau=0.02)
    assert r["f_score"] > 0.9 and r["chamfer_l1"] < 0.03
    assert E.voxel_iou(m.vertices, m.faces, m.vertices, m.faces) > 0.95


def test_shifted_smaller_mesh_scores_worse():
    a = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.6)
    b.apply_translation([0.5, 0, 0])
    r = E.mesh_metrics(a.vertices, a.faces, b.vertices, b.faces, tau=0.02)
    assert r["f_score"] < 0.5


def test_silhouette_iou_perfect_for_matching_render(tmp_path=None):
    class Cam:
        w2c = np.eye(4)
        f = 1.0 / np.tan(np.radians(15))
        cx = cy = 0.0

    cam = Cam()
    cam.w2c = cam.w2c.copy()
    cam.w2c[2, 3] = -3.0
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    from aura_white.forge import softraster as sr

    mask, _d, _t = sr.render_camera(cam, np.asarray(m.vertices), np.asarray(m.faces), 128)
    iou = E.silhouette_iou(m.vertices, m.faces, cam, mask.astype(np.float32), size=128)
    assert iou > 0.98


def test_front_psnr_perfect_match_is_high():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3))
    assert E.front_psnr(img, img) > 60


def test_evaluate_dirs_matches_by_stem(tmp_path):
    pred_dir, gt_dir = tmp_path / "pred", tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    m = trimesh.creation.icosphere(subdivisions=2)
    from aura_white.backends.glb import write_glb_mesh

    write_glb_mesh(pred_dir / "a.glb", m.vertices, m.faces)
    write_glb_mesh(gt_dir / "a.glb", m.vertices, m.faces)
    rep = E.evaluate_dirs(str(pred_dir), str(gt_dir))
    assert rep["n_pairs"] == 1 and rep["n_failed"] == 0
    assert rep["summary"]["f_score"] > 0.9
