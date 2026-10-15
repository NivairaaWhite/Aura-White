"""A frozen benchmark: the same held-out assets, the same seeds, the same metrics - for every checkpoint, every round.

Two kinds of numbers, never mixed:
  * ORACLE metrics compare with the true 3-D shape (chamfer, F-score, back-view agreement).  They are used only to JUDGE
    a model.  They are never used to pick training samples - a real picture-only workflow has no ground truth.
  * VERIFIER metrics (gates.verify) need no ground truth: the score the self-improvement loop is allowed to use.
If the verifier score rises while the oracle numbers do not, the model learned to please the verifier, not to be right;
`compare()` reports both so that is visible.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import geometry as G
from .gates import GateConfig, TargetView, verify
from .model import AuraNative, ViewIn
from .render import fit_camera, render


@dataclass
class BenchItem:
    name: str
    views: List[ViewIn]                  # what the model is shown
    gt_v: np.ndarray                     # the true mesh (normalised frame)
    gt_f: np.ndarray
    gt_c: Optional[np.ndarray]
    expect_solid: bool = True
    meta: Dict = field(default_factory=dict)


def items_from_procedural(ds, n: int, with_views=((0.0, 0.0),), size: int = 64, mesh_res: int = 64) -> List[BenchItem]:
    """Held-out procedural assets: pictures from the stored renders, ground truth regenerated exactly from the seed."""
    from .procedural import make_asset

    out: List[BenchItem] = []
    for i in range(min(n, len(ds))):
        rec, meta = ds[i], ds.meta(i)
        a = make_asset(meta["family"], int(meta["seed"]))
        v, f, c = a.mesh(mesh_res)
        views = []
        for az, el in with_views:
            j = int(np.argmin(np.abs(rec["view_az"] - az) + np.abs(rec["view_el"] - el)))
            views.append(ViewIn(rec["imgs"][j].astype(np.float32) / 255.0, float(az), float(el)))
        out.append(BenchItem(meta["id"], views, v, f, c, expect_solid=(meta["family"] != "blade"), meta=dict(meta)))
    return out


def _sym_chamfer(v1, f1, v2, f2, n=4000, seed=0):
    return G.mesh_chamfer(v1, f1, v2, f2, n=n, seed=seed, taus=(0.03, 0.06))


def _view_agree(v, f, c, gv, gf, gc, az, size=64):
    """Silhouette IoU and colour error between the generated and the true asset seen from `az`, both framed by the TRUE
    object's box.  (The back silhouette is just the mirrored front one, so use az=90 for a depth-sensitive IoU and az=180
    for the colour of the unseen side.)"""
    cam_g = fit_camera(gv, az, 0.0, size)
    a, _, _ = render(v, f, c, cam=cam_g, size=size)                    # generated, framed by the TRUE object's box
    b, _, _ = render(gv, gf, gc, cam=cam_g, size=size)
    ma, mb = a[..., 3] > 0.5, b[..., 3] > 0.5
    u = np.logical_or(ma, mb).sum()
    iou = float(np.logical_and(ma, mb).sum() / u) if u else 1.0
    both = ma & mb
    mae = float(np.abs(a[..., :3][both] - b[..., :3][both]).mean()) if both.sum() > 10 else float("nan")
    return iou, mae


def evaluate(model: AuraNative, items: Sequence[BenchItem], seeds: Sequence[int] = (0, 1), steps: int = 16,
             cfg_scale: float = 3.0, res: int = 48, gate_cfg: Optional[GateConfig] = None, say: Optional[Callable] = None) -> Dict:
    gate_cfg = gate_cfg or GateConfig(size=64)
    rows: List[Dict] = []
    t0 = time.time()
    for k, it in enumerate(items):
        gens = []
        for s in seeds:
            g = model.generate(it.views, n=1, steps=steps, cfg_scale=cfg_scale, seed=int(s), res=res, keep_z=False)[0]
            gens.append(g)
        for s, g in zip(seeds, gens):
            if len(g.faces) < 8:
                rows.append({"item": it.name, "seed": int(s), "empty": True, "chamfer": 0.3, "f06": 0.0, "front_iou": 0.0,
                             "side_iou": 0.0, "back_mae": 0.3, "score": 0.0, "passed": False})
                continue
            tv = [TargetView(np.asarray(v.image, np.float32), v.az, v.el, f"v{i}") for i, v in enumerate(it.views)]
            ver = verify(g.verts, g.faces, g.colours, tv, it.expect_solid, cfg=gate_cfg)
            ch = _sym_chamfer(g.verts, g.faces, it.gt_v, it.gt_f)
            fi, fm = _view_agree(g.verts, g.faces, g.colours, it.gt_v, it.gt_f, it.gt_c, 0.0)
            si, _ = _view_agree(g.verts, g.faces, g.colours, it.gt_v, it.gt_f, it.gt_c, 90.0)
            _, bm = _view_agree(g.verts, g.faces, g.colours, it.gt_v, it.gt_f, it.gt_c, 180.0)
            rows.append({"item": it.name, "seed": int(s), "empty": False, "chamfer": ch["chamfer"], "f06": ch["f0.06"],
                         "front_iou": fi, "front_mae": fm, "side_iou": si, "back_mae": bm, "score": ver.score,
                         "passed": bool(ver.passed), "failing": ver.failing})
        # diversity: do different seeds give different shapes?
        if len(gens) >= 2 and len(gens[0].faces) > 8 and len(gens[1].faces) > 8:
            d = _sym_chamfer(gens[0].verts, gens[0].faces, gens[1].verts, gens[1].faces, n=2000)["chamfer"]
            rows.append({"item": it.name, "diversity": d})
        if say and (k + 1) % 10 == 0:
            say(f"    bench {k + 1}/{len(items)} ({time.time() - t0:.0f}s)")
    main = [r for r in rows if "chamfer" in r]
    div = [r["diversity"] for r in rows if "diversity" in r]

    def m(key):
        a = np.array([r[key] for r in main if key in r and np.isfinite(r[key])], np.float64)
        return (float(a.mean()), float(a.std() / np.sqrt(max(len(a), 1)))) if len(a) else (float("nan"), float("nan"))
    summ = {"n_items": len(items), "n_samples": len(main), "seconds": round(time.time() - t0, 1)}
    for key in ("chamfer", "f06", "front_iou", "side_iou", "back_mae", "score"):
        mean, se = m(key)
        summ[key], summ[key + "_se"] = mean, se
    summ["pass_rate"] = float(np.mean([r["passed"] for r in main])) if main else 0.0
    summ["empty_rate"] = float(np.mean([r["empty"] for r in main])) if main else 1.0
    summ["diversity"] = float(np.mean(div)) if div else float("nan")
    return {"summary": summ, "rows": rows}


def compare(a: Dict, b: Dict, a_name: str = "A", b_name: str = "B") -> str:
    """A small text table: oracle numbers first, verifier numbers after."""
    sa, sb = a["summary"], b["summary"]
    lines = [f"{'metric':14s}{a_name:>14s}{b_name:>14s}   (+ = better for B)"]
    for key, better in (("chamfer", -1), ("f06", 1), ("side_iou", 1), ("back_mae", -1), ("front_iou", 1), ("score", 1), ("pass_rate", 1), ("diversity", 1)):
        x, y = sa.get(key, float("nan")), sb.get(key, float("nan"))
        lines.append(f"{key:14s}{x:14.4f}{y:14.4f}   {better * (y - x):+.4f}")
    return "\n".join(lines)


def save_report(res: Dict, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
