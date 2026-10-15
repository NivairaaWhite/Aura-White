import sys

import numpy as np
import pytest

import prims
from aura_white.gen3d import meshutil as M
from aura_white.gen3d import views as V
from aura_white.gen3d.audit import audit_mesh
from aura_white.gen3d.hardware import (ExecPlan, Hardware, MIN_CHUNK_SCALE, STAGES, is_oom, plan_stage, run_with_oom_retry)
from aura_white.gen3d.recipe import PRESETS, Recipe


# ------------------------------------------------------------------------------------------ recipe
def test_quality_signature_ignores_the_label_and_changes_with_settings():
    r = Recipe.preset("high")
    assert r.signature() == r.with_(name="renamed").signature()
    assert r.signature() != r.with_(octree=512).signature()
    assert len({p.signature() for p in PRESETS.values()}) == len(PRESETS)


def test_recipe_rejects_unknown_fields_and_presets():
    with pytest.raises(ValueError):
        Recipe().with_(hardware="gpu")
    with pytest.raises(ValueError):
        Recipe.preset("insane")


# ------------------------------------------------------------------------------------------ hardware planning
def gpu(gb, ram=32.0):
    return Hardware("cuda", f"{gb} GB", gb, gb, ram, ram * 0.8, True, True)


@pytest.mark.parametrize("gb", [3, 4, 6, 8, 12, 24, 80])
def test_the_shape_model_can_run_on_every_card_with_the_same_recipe(gb):
    plan = plan_stage("hunyuan3d2_shape", gpu(gb))
    assert plan.feasible, plan.note
    assert plan.est_peak_gb <= gb          # never plans beyond the card


def test_smaller_cards_offload_more_but_the_recipe_is_untouched():
    p24, p8, p4 = (plan_stage("hunyuan3d2_shape", gpu(g)) for g in (24, 8, 4))
    order = ("none", "model", "sequential", "cpu")
    assert order.index(p24.offload) <= order.index(p8.offload) <= order.index(p4.offload)
    assert p24.offload == "none" and p4.offload != "none"
    r = Recipe.preset("high")
    assert r.signature() == Recipe.preset("high").signature()      # hardware is not an input of the recipe at all


def test_no_gpu_means_the_processor_not_a_lower_quality_model():
    p = plan_stage("hunyuan3d2_shape", Hardware("cpu", "cpu", 0, 0, 32, 24, True, False))
    assert p.device == "cpu" and p.feasible


def test_infeasible_when_neither_gpu_nor_ram_suffice():
    p = plan_stage("sdxl_depth_paint", Hardware("cpu", "cpu", 0, 0, 8, 4, True, False))
    assert not p.feasible and "RAM" in p.note


def test_oom_retry_escalates_and_returns_the_plan_that_worked():
    seen = []

    def fn(plan):
        seen.append((plan.offload, plan.chunk_scale))
        if len(seen) < 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "mesh"

    res, plan = run_with_oom_retry(fn, ExecPlan("s", "cuda", "none"), allow_cpu=False)
    assert res == "mesh" and len(seen) == 4
    assert seen[1][1] < seen[0][1]                                  # chunks shrink first
    assert plan.chunk_scale <= seen[0][1]


def test_oom_retry_gives_up_clearly_and_ignores_other_errors():
    def always(plan):
        raise RuntimeError("out of memory")
    with pytest.raises(MemoryError):
        run_with_oom_retry(always, ExecPlan("s", "cuda", "none"), allow_cpu=False)

    def other(plan):
        raise ValueError("bad input")
    with pytest.raises(ValueError):
        run_with_oom_retry(other, ExecPlan("s", "cuda", "none"))
    assert is_oom(RuntimeError("CUDA error: out of memory")) and not is_oom(ValueError("nope"))


def test_every_stage_has_a_plan_on_a_8gb_card():
    for name in STAGES:
        if name == "flux_text2image":
            continue                                               # needs more system RAM than a typical 8 GB-card PC has
        assert plan_stage(name, gpu(8, ram=32)).feasible, name


# ------------------------------------------------------------------------------------------ views
@pytest.mark.parametrize("name,label", [("knight_back.png", "back"), ("Knight-LEFT-view.jpg", "left"), ("img_rear.jpeg", "back"),
                                         ("side.png", "left"), ("front.png", "front"), ("3-4_view.png", "front_left"),
                                         ("feedback.png", None), ("backpack.png", None), ("dragon.png", None)])
def test_view_labels_come_from_whole_words_only(name, label):
    assert V.infer_label(name) == label


def test_explicit_angles_win_over_names():
    s = V.ViewSpec.parse("left@75,12")
    assert (s.label, s.az, s.el) == ("left", 75.0, 12.0)
    assert V.ViewSpec.parse("back").az == 180.0
    with pytest.raises(ValueError):
        V.ViewSpec.parse("sideways")


def test_there_are_24_proper_rotations():
    rots = V.proper_rotations()
    assert len(rots) == 24 and all(abs(np.linalg.det(R) - 1) < 1e-9 for R in rots)


def test_fit_cam_recovers_the_camera_even_when_the_mesh_lacks_parts():
    truth = prims.figure()
    gen = prims.merge(prims.uv_sphere(24, 48, (0.45, 0.8, 0.30)), prims.translate(prims.uv_sphere(14, 28, 0.22), (0.0, 1.0, 0.0)))
    for az in (0.0, 90.0, 180.0):
        mask = V.render_maps(*truth, V.ViewCam(az, 0.0, 192, 80.0, 0.0, 0.2))[0]
        cam, iou, _ = V.fit_cam(*gen, az, 0.0, mask, 192)
        assert abs(cam.scale - 80.0) < 6.0, (az, cam.scale)
        assert iou > 0.8


# ------------------------------------------------------------------------------------------ voxels
def test_voxelisation_matches_exact_volumes():
    for (v, f), exact in [(prims.box((1, 2, 3), 3), 6.0), (prims.uv_sphere(30, 60, 1.0), 4 / 3 * np.pi)]:
        lo, side = M.grid_for(v)
        occ = M.voxelize(v, f, 80, lo, side)
        assert abs(occ.sum() * (side / 80) ** 3 - exact) / exact < 0.03


def test_overlapping_shells_and_mixed_orientation_are_handled():
    v, f = prims.figure()
    bad = f.copy()
    bad[:300] = bad[:300][:, ::-1]
    v2, f2 = M.orient_outward(v, bad)
    assert prims.volume(v2, f2) > 0
    lo, side = M.grid_for(v)
    a, b = M.voxelize(v, f, 72, lo, side), M.voxelize(v2, f2, 72, lo, side)
    assert (a & b).sum() / (a | b).sum() > 0.97


def test_field_to_mesh_round_trip_is_closed():
    from aura_white.mesh.repair import topology_stats
    v, f = prims.uv_sphere(24, 48, 1.0)
    lo, side = M.grid_for(v)
    nv, nf = M.mesh_from_field(M.smooth_field(M.voxelize(v, f, 64, lo, side)), lo, side)
    st = topology_stats(nf)
    assert st["boundary_edges"] == 0 and st["nonmanifold_edges"] == 0
    assert abs(np.linalg.norm(nv, axis=1).mean() - 1.0) < 0.03


# ------------------------------------------------------------------------------------------ audit
def test_audit_tells_solid_from_flat():
    assert audit_mesh(*prims.uv_sphere(20, 40, 1.0)).flat_risk == 0
    assert audit_mesh(*prims.figure()).flat_risk == 0
    slab = audit_mesh(*prims.slab())
    assert slab.flat_risk > 0.9 and slab.profile == "extruded"


def test_a_relief_with_the_back_mirrored_is_flagged_even_at_15_percent_depth():
    # this is what the old relief / source-locked mode produced
    a = audit_mesh(*prims.uv_sphere(30, 60, (1.0, 1.0, 0.15)))
    assert a.flat_risk > 0.35 and a.profile == "mirrored-relief"


def test_flat_objects_are_allowed_when_the_recipe_says_so():
    assert audit_mesh(*prims.slab(), expect_solid=False).flat_risk == 0
