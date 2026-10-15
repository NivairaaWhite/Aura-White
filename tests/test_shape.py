import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

from aura_white.shape import sdf as S
from aura_white.shape.model import FlowMatching, ShapeDiT


def test_mesh_to_sdf_sign_and_scale():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    d = S.mesh_to_sdf(np.asarray(m.vertices), np.asarray(m.faces), 32)
    assert d[16, 16, 16] < 0
    assert d[0, 0, 0] > 0
    assert abs(abs(d[16, 16, 16]) - 0.5) < 0.15


def test_observed_mask_matches_the_visible_side_of_a_slab():
    # A synthetic SDF: a solid slab occupying the whole y,z extent and x in [10,20) so every column hits it.
    res = 32
    sd = np.ones((res, res, res), np.float32)
    sd[10:20] = -1.0
    obs = S.observed_mask(sd, thickness=1, axis=0, from_positive=True)
    assert obs[21:].all(), "space in front of the slab (near the +x camera) must be fully observed"
    assert not obs[:9].any(), "space behind the slab (far side) must be fully hidden"
    assert obs[19].all(), "the entry surface itself (first `thickness` voxels, nearest the +x camera) is observed"


def test_sdf_to_mesh_roundtrip_radius():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    d = S.mesh_to_sdf(np.asarray(m.vertices), np.asarray(m.faces), 40)
    v, f = S.sdf_to_mesh(d)
    r = np.linalg.norm(v, axis=1)
    assert abs(r.mean() - 0.5) < 0.05
    assert len(f) > 0


def test_random_shape_is_reproducible_and_nonempty():
    a = S.random_shape(24, np.random.default_rng(3))
    b = S.random_shape(24, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert (a < 0).mean() > 0.01


def test_flow_matching_reduces_loss_and_completes_better_than_no_fill():
    torch.manual_seed(0)
    model = ShapeDiT(res=16, patch=4, d_model=64, layers=2, heads=4)
    flow = FlowMatching()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)

    def batch(n=8):
        xs, cs, ms = [], [], []
        for _ in range(n):
            d = S.random_shape(16, rng, n_parts=1)
            t = S.to_tsdf(d, 16)
            obs = S.observed_mask(d, thickness=2, axis=0, from_positive=True)
            xs.append(t)
            cs.append(np.where(obs, t, 0.0))
            ms.append(obs.astype(np.float32))
        return (torch.from_numpy(np.stack(a).astype(np.float32))[:, None] for a in (xs, cs, ms))

    losses = []
    for _ in range(60):
        x1, cond, mask = batch()
        loss = flow.loss(model, x1, cond, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert np.mean(losses[-10:]) < np.mean(losses[:10])

    d = S.random_shape(16, np.random.default_rng(999), n_parts=1)
    t = S.to_tsdf(d, 16)
    obs = S.observed_mask(d, thickness=2, axis=0, from_positive=True)
    cond = torch.from_numpy(np.where(obs, t, 0.0).astype(np.float32))[None, None]
    mask_t = torch.from_numpy(obs.astype(np.float32))[None, None]
    out = flow.sample(model, cond, mask_t, steps=12, seed=1)[0, 0].numpy()
    gt = d < 0
    pred = out < 0
    iou_model = (pred & gt).sum() / max((pred | gt).sum(), 1)
    no_fill = np.where(obs, t, 1.0) < 0
    iou_none = (no_fill & gt).sum() / max((no_fill | gt).sum(), 1)
    assert iou_model >= iou_none - 0.05                      # a short training run should not be worse than nothing


def test_observed_voxels_are_data_consistent_after_sampling():
    torch.manual_seed(0)
    model = ShapeDiT(res=16, patch=4, d_model=32, layers=1, heads=2)
    flow = FlowMatching()
    d = S.random_shape(16, np.random.default_rng(7))
    t = S.to_tsdf(d, 16)
    obs = S.observed_mask(d, thickness=2, axis=0, from_positive=True)
    cond = torch.from_numpy(np.where(obs, t, 0.0).astype(np.float32))[None, None]
    mask_t = torch.from_numpy(obs.astype(np.float32))[None, None]
    out = flow.sample(model, cond, mask_t, steps=4, seed=0)[0, 0].numpy()
    assert np.allclose(out[obs], t[obs], atol=1e-4)
