"""Quality metrics for 3-D generation (NumPy + SciPy; no trimesh, no torch).

  chamfer / f_score / normal_consistency   mesh vs reference mesh (surface samples, KD-tree)
  voxel_iou                                 occupancy IoU through `shape.sdf.mesh_to_sdf`
  silhouette_iou / front_psnr               a textured or plain mesh rendered in the artwork's camera vs the artwork
  evaluate_dirs                             batch driver over matching file names
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np


def _samples(v: np.ndarray, f: np.ndarray, n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tri = np.asarray(v, np.float64)[f]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    pts = a[:, None] * tri[idx, 0] + b[:, None] * tri[idx, 1] + c[:, None] * tri[idx, 2]
    nrm = cr[idx] / np.maximum(np.linalg.norm(cr[idx], axis=1, keepdims=True), 1e-30)
    return pts, nrm


def normalize_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit both meshes into the same unit box using the reference's bounds (so scale differences are visible)."""
    c = 0.5 * (b.min(0) + b.max(0))
    s = 1.0 / max(float(np.abs(b - c).max()), 1e-12)
    return (a - c) * s, (b - c) * s


def mesh_metrics(pv, pf, gv, gf, tau: float = 0.02, n: int = 30000, normalize: bool = True) -> Dict[str, float]:
    from scipy.spatial import cKDTree

    pv, gv = np.asarray(pv, np.float64), np.asarray(gv, np.float64)
    if normalize:
        pv, gv = normalize_pair(pv, gv)
    pa, na = _samples(pv, pf, n, 0)
    pb, nb = _samples(gv, gf, n, 1)
    ta, tb = cKDTree(pa), cKDTree(pb)
    d_ab, i_ab = tb.query(pa, workers=-1)          # pred -> gt (accuracy)
    d_ba, _ = ta.query(pb, workers=-1)             # gt -> pred (completeness)
    prec, rec = float((d_ab < tau).mean()), float((d_ba < tau).mean())
    return {"chamfer_l1": float(d_ab.mean() + d_ba.mean()), "chamfer_l2": float((d_ab ** 2).mean() + (d_ba ** 2).mean()),
            "accuracy": float(d_ab.mean()), "completeness": float(d_ba.mean()),
            "f_score": 2 * prec * rec / max(prec + rec, 1e-12), "precision": prec, "recall": rec, "tau": tau,
            "normal_consistency": float(np.abs((na * nb[i_ab]).sum(1)).mean())}


def voxel_iou(pv, pf, gv, gf, res: int = 48) -> float:
    from .shape.sdf import mesh_to_sdf

    pv, gv = normalize_pair(np.asarray(pv, np.float64), np.asarray(gv, np.float64))
    s = 0.85
    a = mesh_to_sdf(pv * s, pf, res) < 0
    b = mesh_to_sdf(gv * s, gf, res) < 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def silhouette_iou(verts, faces, camera, ref_alpha: np.ndarray, size: int = 512) -> float:
    """IoU between the mesh silhouette in `camera` and the artwork alpha (>= 0.5)."""
    from PIL import Image

    from .forge import softraster as sr

    a = np.asarray(Image.fromarray((ref_alpha * 255 + 0.5).astype(np.uint8)).resize((size, size), Image.BILINEAR)) >= 128
    m, _d, _t = sr.render_camera(camera, np.asarray(verts, np.float64), np.asarray(faces, np.int64), size)
    return float((m & a).sum() / max((m | a).sum(), 1))


def front_psnr(render_rgb: np.ndarray, ref_rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """PSNR between a render and the artwork over the artwork's foreground (both float [0,1])."""
    d = (np.asarray(render_rgb, np.float64) - np.asarray(ref_rgb, np.float64)) ** 2
    if mask is not None and mask.any():
        d = d[mask]
    mse = float(d.mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def load_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    p = path.lower()
    if p.endswith((".glb", ".gltf")):
        from .backends.glb import read_glb

        return read_glb(path)
    if p.endswith(".obj"):
        from .shape.train import _read_obj

        return _read_obj(path)
    raise ValueError(f"unsupported mesh format: {path} (use .glb or .obj)")


def evaluate_dirs(pred_dir: str, gt_dir: str, tau: float = 0.02, n: int = 30000) -> dict:
    def index(d):
        out = {}
        for dp, _, fs in os.walk(d):
            for f in fs:
                if f.lower().endswith((".glb", ".gltf", ".obj")):
                    out[os.path.splitext(f)[0]] = os.path.join(dp, f)
        return out

    P, G = index(pred_dir), index(gt_dir)
    rows = []
    for k in sorted(set(P) & set(G)):
        try:
            pv, pf = load_mesh(P[k])
            gv, gf = load_mesh(G[k])
            r = mesh_metrics(pv, pf, gv, gf, tau=tau, n=n)
            r["voxel_iou"] = voxel_iou(pv, pf, gv, gf)
            r["name"] = k
            rows.append(r)
        except Exception as e:                       # one bad file must not kill the batch
            rows.append({"name": k, "error": f"{type(e).__name__}: {e}"})
    ok = [r for r in rows if "error" not in r]
    summary = {m: float(np.mean([r[m] for r in ok])) for m in ("chamfer_l1", "f_score", "normal_consistency", "voxel_iou")} if ok else {}
    return {"n_pairs": len(ok), "n_failed": len(rows) - len(ok), "summary": summary, "pairs": rows}


def save_report(rep: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
