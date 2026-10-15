"""Tangent frames and tangent-space normal-map baking (high-poly detail -> low-poly UV atlas).

Conventions follow glTF: red = +u, green = image-up (= -v, because v grows downwards), blue = along the surface
normal; bitangent = cross(normal, tangent) * w."""
from __future__ import annotations

from typing import Optional

import numpy as np


def tangent_frames(verts: np.ndarray, faces: np.ndarray, uv: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Per-vertex tangents (V,4): xyz orthogonal to the normal, w = handedness (+-1)."""
    v = verts.astype(np.float64)
    e1, e2 = v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]
    d1, d2 = uv[faces[:, 1]] - uv[faces[:, 0]], uv[faces[:, 2]] - uv[faces[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-14, 1.0 / np.where(np.abs(det) > 1e-14, det, 1.0), 0.0)[:, None]
    tu = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r                # dP/du
    tv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r                # dP/dv
    area = np.linalg.norm(np.cross(e1, e2), axis=1)[:, None]
    T = np.zeros_like(v)
    B = np.zeros_like(v)
    for k in range(3):
        np.add.at(T, faces[:, k], tu * area)
        np.add.at(B, faces[:, k], -tv * area)                   # image-up direction
    n = normals.astype(np.float64)
    T = T - n * (n * T).sum(1, keepdims=True)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    empty = np.linalg.norm(T, axis=1) < 0.5                     # vertices without a usable UV gradient
    if empty.any():
        alt = np.cross(n, np.array([0.0, 0.0, 1.0]))
        alt[np.linalg.norm(alt, axis=1) < 1e-6] = np.array([1.0, 0.0, 0.0])
        T[empty] = alt[empty] / np.linalg.norm(alt[empty], axis=1, keepdims=True)
    w = np.where((np.cross(n, T) * B).sum(1) < 0, -1.0, 1.0)
    return np.concatenate([T, w[:, None]], axis=1).astype(np.float32)


def bake_normal_map(raster: "Raster", hi_verts: np.ndarray, hi_faces: np.ndarray, lo_verts: np.ndarray,
                    lo_faces: np.ndarray, lo_uv: np.ndarray, lo_normals: np.ndarray, size: int,
                    samples_per_face: int = 3, min_align: float = 0.25, gutter: Optional[int] = None,
                    progress=None) -> Optional[np.ndarray]:
    """Returns an (size,size,3) uint8 tangent-space normal map, or None when scipy is unavailable."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    import torch                                       # lazy: tangent_frames() above must stay importable without torch
    import torch.nn.functional as F

    from ..mesh.cleanup import vertex_normals
    from ..raster import interpolate
    from .fill import dilate_gutters

    say = progress or (lambda *_: None)
    dev = raster.device
    # dense point sample of the high-poly surface with its smooth normals
    hv = hi_verts.astype(np.float64)
    hn = vertex_normals(hi_verts, hi_faces).astype(np.float64)
    rng = np.random.default_rng(0)
    b = rng.dirichlet((1.0, 1.0, 1.0), size=(len(hi_faces), samples_per_face))          # (F,S,3)
    pts = (hv[hi_faces][:, None] * b[..., None]).sum(2).reshape(-1, 3)
    nrm = (hn[hi_faces][:, None] * b[..., None]).sum(2).reshape(-1, 3)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    say("building surface search tree")
    tree = cKDTree(pts)

    tang = tangent_frames(lo_verts, lo_faces, lo_uv, lo_normals)
    V = torch.as_tensor(lo_verts, dtype=torch.float32, device=dev)
    Fc = torch.as_tensor(lo_faces, dtype=torch.long, device=dev)
    ur = raster.rasterize_uv(torch.as_tensor(lo_uv, dtype=torch.float32, device=dev), Fc, size)
    hit = ur.mask
    P = interpolate(V, Fc, ur)[hit]
    N = F.normalize(interpolate(torch.as_tensor(lo_normals, dtype=torch.float32, device=dev), Fc, ur)[hit], dim=-1)
    T = interpolate(torch.as_tensor(tang[:, :3], device=dev), Fc, ur)[hit]
    W = interpolate(torch.as_tensor(tang[:, 3:4], device=dev), Fc, ur)[hit][:, 0]
    T = F.normalize(T - N * (N * T).sum(-1, keepdim=True), dim=-1)
    B = torch.cross(N, T, dim=-1) * torch.sign(W + 1e-9).unsqueeze(-1)

    say("looking up high-poly normals")
    k = 4
    _, idx = tree.query(P.cpu().numpy(), k=k, workers=-1)
    cand = torch.as_tensor(nrm[idx], dtype=torch.float32, device=dev)                 # (n,k,3)
    align = (cand * N.unsqueeze(1)).sum(-1)                                            # (n,k)
    best = align.argmax(1)
    chosen = cand[torch.arange(len(best)), best]
    ok = align[torch.arange(len(best)), best] > min_align
    chosen = torch.where(ok.unsqueeze(-1), chosen, N)
    tn = torch.stack([(chosen * T).sum(-1), (chosen * B).sum(-1), (chosen * N).sum(-1)], -1)
    tn = F.normalize(tn, dim=-1)
    rgb = torch.zeros(size, size, 3, device=dev)
    rgb[..., 2] = 1.0
    rgb[..., :2] = 0.0
    flat = torch.zeros(size, size, 3, device=dev)
    flat[hit] = tn * 0.5 + 0.5
    flat[~hit] = torch.tensor([0.5, 0.5, 1.0], device=dev)
    flat = dilate_gutters(flat, hit, gutter or max(6, size // 128))
    return (flat.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
