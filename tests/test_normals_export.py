import struct

import numpy as np
import pytest
import torch
import trimesh

from aura_white.mesh import vertex_normals
from aura_white.pipeline import orient_vertices
from aura_white.mesh.textured import (glb_textured_bytes, to_up_axis, view_bytes_textured, write_obj_textured)
from aura_white.raster import get_raster, interpolate
from aura_white.texture import bake_normal_map, decimate, tangent_frames, unwrap

pytest.importorskip("scipy")


def bumpy_pair():
    hi = trimesh.creation.icosphere(5)
    p = np.asarray(hi.vertices)
    disp = 0.06 * np.sin(9 * p[:, 0]) * np.sin(9 * p[:, 1]) * np.sin(9 * p[:, 2] + 1)
    hv, hf = (p * (1 + disp)[:, None]).astype(np.float32), np.asarray(hi.faces)
    lv, lf, _ = decimate(hv, hf, 2000)
    nv, nf, uv, vm, _ = unwrap(lv, lf, 512, "builtin")
    return hv, hf, nv, nf, uv, vertex_normals(lv, lf)[vm]


def test_tangent_frames_are_orthonormal():
    _, _, nv, nf, uv, n = bumpy_pair()
    t = tangent_frames(nv, nf, uv, n)
    assert np.abs((t[:, :3] * n).sum(1)).max() < 1e-3
    assert np.allclose(np.linalg.norm(t[:, :3], axis=1), 1, atol=1e-3)
    assert set(np.unique(t[:, 3])) <= {-1.0, 1.0}


def test_normal_map_moves_shading_towards_the_high_poly_surface():
    hv, hf, nv, nf, uv, n = bumpy_pair()
    R = get_raster("cpu")
    nm = bake_normal_map(R, hv, hf, nv, nf, uv, n, 256)
    assert nm.shape == (256, 256, 3) and nm.dtype == np.uint8
    V, Fc = torch.tensor(nv), torch.tensor(nf)
    ur = R.rasterize_uv(torch.tensor(uv), Fc, 256)
    hit = ur.mask
    P = interpolate(V, Fc, ur)[hit].numpy()
    N = interpolate(torch.tensor(n), Fc, ur)[hit].numpy()
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    tg = tangent_frames(nv, nf, uv, n)
    T = interpolate(torch.tensor(tg[:, :3]), Fc, ur)[hit].numpy()
    W = interpolate(torch.tensor(tg[:, 3:4]), Fc, ur)[hit].numpy()[:, 0]
    T = T - N * (N * T).sum(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    B = np.cross(N, T) * np.sign(W)[:, None]
    rgb = nm[hit.numpy()].astype(np.float32) / 255 * 2 - 1
    Nw = rgb[:, :1] * T + rgb[:, 1:2] * B + rgb[:, 2:] * N
    Nw /= np.linalg.norm(Nw, axis=1, keepdims=True)
    from scipy.spatial import cKDTree

    truth = vertex_normals(hv, hf)[cKDTree(hv).query(P)[1]]
    ang = lambda a, b: np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).mean()   # noqa: E731
    assert ang(Nw, truth) < 0.6 * ang(N, truth)


def test_glb_roundtrip_with_textures():
    m = trimesh.creation.icosphere(2)
    v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces)
    nv, nf, uv, vm, _ = unwrap(v, f, 128, "builtin")
    alb = (np.random.default_rng(0).random((128, 128, 3)) * 255).astype(np.uint8)
    nm = np.full((128, 128, 3), (128, 128, 255), np.uint8)
    tg = tangent_frames(nv, nf, uv, vertex_normals(v, f)[vm])
    blob = glb_textured_bytes(to_up_axis(nv), nf, uv, alb, normal_map=nm, tangents=tg)
    assert blob[:4] == b"glTF" and struct.unpack("<I", blob[8:12])[0] == len(blob)
    g = list(trimesh.load(trimesh.util.wrap_as_stream(blob), file_type="glb").geometry.values())[0]
    assert len(g.vertices) == len(nv) and len(g.faces) == len(nf)
    assert np.allclose(g.visual.uv[:, 0], uv[:, 0], atol=1e-6) and np.allclose(g.visual.uv[:, 1], 1 - uv[:, 1], atol=1e-6)
    assert g.visual.material.baseColorTexture.size == (128, 128)
    assert g.visual.material.normalTexture is not None
    assert np.array_equal(np.asarray(g.visual.material.baseColorTexture.convert("RGB")), alb)


def test_obj_export_and_viewer_blob(tmp_path):
    m = trimesh.creation.icosphere(1)
    v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces)
    nv, nf, uv, _, _ = unwrap(v, f, 64, "builtin")
    alb = np.full((64, 64, 3), 200, np.uint8)
    p = write_obj_textured(tmp_path / "a.obj", nv, nf, uv, alb, normal_map=alb)
    assert (tmp_path / "a.mtl").exists() and (tmp_path / "a_albedo.png").exists() and (tmp_path / "a_normal.png").exists()
    assert "map_Kd a_albedo.png" in (tmp_path / "a.mtl").read_text()
    o = trimesh.load(p, process=False)
    assert len(o.faces) == len(nf)
    blob = view_bytes_textured(nv, nf, uv)
    assert blob[:4] == b"AWV2"
    n_v, n_f = struct.unpack("<II", blob[4:12])
    assert len(blob) == 12 + n_v * (12 + 12 + 8) + n_f * 12


def test_up_axis_matches_the_existing_exporter():
    v = np.random.default_rng(1).random((20, 3)).astype(np.float32)
    assert np.allclose(to_up_axis(v, "y"), orient_vertices(v, "y"))
    assert np.allclose(to_up_axis(v, "z"), v)
