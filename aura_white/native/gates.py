"""Quality gates: the verifier that decides what Aura may publish, what a teacher's output is worth, and which of Aura's
own samples are allowed to teach Aura.

Same six gates as the project brief (numpy + scipy only, deterministic, no model inside):
  G1 back_distinct   the back is not a copy / mirror of the front (geometry and colour)
  G2 real_depth      not a slab / relief; no stair-stepped surface
  G3 colour          surface colours are real: finite, not flat black / magenta placeholders, not a single flat tone
  G4 mesh            closed, manifold, one main piece, within the triangle budget
  G5 fidelity        re-rendered from the picture's own camera it matches the picture (silhouette IoU + colour)
  G6 file            the GLB loads back, indices in range, no NaN (`check_glb`)

`verify()` returns a `Verdict`: a score in [0,1] (used as the reward in the self-improvement loop), a pass flag (used as
the publish gate) and every number behind them (used in reports).  The thresholds are *starting values* - run
`calibrate()` on a folder of assets you consider good and read the distribution before trusting a pass rate.

Honest limits: these gates measure structure and agreement with the pictures you gave.  They do not measure beauty,
and they cannot see a wrong back side that is internally consistent - only a back picture or a human can.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .render import fit_camera, render


@dataclass
class GateConfig:
    size: int = 96
    min_depth_ratio: float = 0.08          # thinnest / longest bounding-box side for an object that must be solid
    max_sharp_edge_frac: float = 0.05      # share of adjacent face pairs bending more than `sharp_deg` (stair-steps)
    sharp_deg: float = 75.0
    mirror_ssim_fail: float = 0.985        # colour SSIM(back, mirrored front) above this = a pasted copy (when textured)
    mirror_geom_fail: float = 0.995
    min_view_iou: float = 0.80             # silhouette agreement with a supplied picture, per view
    face_budget: int = 400_000
    pass_score: float = 0.70
    weights: Dict[str, float] = field(default_factory=lambda: {"fidelity": 0.30, "mesh": 0.20, "depth": 0.20,
                                                                "back": 0.15, "colour": 0.15})


@dataclass
class TargetView:
    image: np.ndarray            # (S,S,4) float, as produced by render.prep_picture (rgb over grey, alpha mask)
    az: float = 0.0
    el: float = 0.0
    name: str = "front"


@dataclass
class Verdict:
    passed: bool
    score: float
    gates: Dict[str, Dict]
    failing: List[str]

    def as_dict(self) -> Dict:
        return {"passed": self.passed, "score": round(float(self.score), 4), "failing": self.failing,
                "gates": {k: {kk: (round(float(vv), 4) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in g.items()}
                          for k, g in self.gates.items()}}


# ------------------------------------------------------------------------------------------------ helpers
def ssim(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """Mean structural similarity of two grey images in [0,1] (uniform window; no external dependency)."""
    from scipy.ndimage import uniform_filter

    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = uniform_filter(a, win), uniform_filter(b, win)
    saa, sbb, sab = uniform_filter(a * a, win) - ma * ma, uniform_filter(b * b, win) - mb * mb, uniform_filter(a * b, win) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma * ma + mb * mb + c1) * (saa + sbb + c2))
    return float(s.mean())


def sharp_edge_fraction(v: np.ndarray, f: np.ndarray, deg: float = 75.0) -> float:
    """Share of interior edges whose two faces meet at more than `deg` degrees.  A smooth surface is near 0; a voxel
    stair-case (the look of a depth-map relief or a low-resolution voxel stack seen from the side) is large."""
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    if len(f) < 4:
        return 1.0
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fid = np.tile(np.arange(len(f)), 3)
    key = np.sort(e, 1)
    k = key[:, 0] * (int(v.shape[0]) + 1) + key[:, 1]
    o = np.argsort(k, kind="stable")
    k, fid = k[o], fid[o]
    same = k[1:] == k[:-1]
    a, b = fid[:-1][same], fid[1:][same]
    if len(a) == 0:
        return 0.0
    cosang = (n[a] * n[b]).sum(1)
    return float((cosang < np.cos(np.radians(deg))).mean())


def _soft(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


# ------------------------------------------------------------------------------------------------ the gates
def gate_mesh(v, f, cfg: GateConfig, expect_solid=True) -> Dict:
    from ..gen3d.audit import audit_mesh

    a = audit_mesh(v, f, min_depth_ratio=cfg.min_depth_ratio * 2.75, expect_solid=False)
    over = len(f) > cfg.face_budget
    score = a.topology_score * (0.5 if over else 1.0)
    return {"score": score, "ok": a.topology_score >= 0.5 and not over, "faces": a.faces, "watertight": a.watertight,
            "boundary_edges": a.boundary_edges, "nonmanifold_edges": a.nonmanifold_edges, "components": a.components,
            "largest_share": a.largest_share, "over_budget": over, "_audit": a}


def gate_depth(v, f, cfg: GateConfig, expect_solid=True, audit=None) -> Dict:
    from ..gen3d.audit import audit_mesh

    ext = np.sort(np.asarray(v, np.float64).max(0) - np.asarray(v, np.float64).min(0))
    ratio = float(ext[0] / max(ext[2], 1e-9))
    a = audit or audit_mesh(v, f, min_depth_ratio=cfg.min_depth_ratio * 2.75, expect_solid=False)
    sharp = sharp_edge_fraction(v, f, cfg.sharp_deg)
    slab = a.profile in ("extruded", "mirrored-relief")
    flat = ratio < cfg.min_depth_ratio
    s_flat = 1.0 if not expect_solid else _soft(ratio, cfg.min_depth_ratio * 0.5, cfg.min_depth_ratio * 1.5)
    s_stair = 1.0 - _soft(sharp, cfg.max_sharp_edge_frac, cfg.max_sharp_edge_frac * 3.0)
    s_slab = 0.0 if (slab and expect_solid) else 1.0
    score = min(s_flat, s_slab) * (0.4 + 0.6 * s_stair)
    ok = not (expect_solid and (flat or slab)) and sharp <= cfg.max_sharp_edge_frac * 2.0
    return {"score": score, "ok": bool(ok), "thin_ratio": ratio, "sharp_edge_frac": sharp, "profile": a.profile,
            "mirror_corr": a.mirror_corr, "thickness_cv": a.thickness_cv}


def gate_back(v, f, colours, cfg: GateConfig, symmetric_ok: bool = False) -> Dict:
    """Render the front and the back with the SAME framing; the back, flipped, is compared with the front."""
    size = cfg.size
    cam_f = fit_camera(v, 0.0, 0.0, size)
    cam_b = cam_f.with_(az=180.0, ox=-cam_f.ox)
    rf, df, _ = render(v, f, colours, cam=cam_f, size=size)
    rb, db, _ = render(v, f, colours, cam=cam_b, size=size)
    rb_m, db_m = rb[:, ::-1], db[:, ::-1]
    both = (rf[..., 3] > 0.5) & (rb_m[..., 3] > 0.5)
    if both.sum() < 30:
        return {"score": 1.0, "ok": True, "note": "views barely overlap", "geom_ssim": float("nan"), "colour_ssim": float("nan")}
    zf = np.nan_to_num(df, nan=0.0)
    zb = np.nan_to_num(-db_m, nan=0.0)
    lo, hi = float(np.nanmin(df)), float(np.nanmax(df))
    rng_z = max(hi - lo, 1e-9)
    # geometry: does the back surface follow the front surface mirrored (thickness constant = slab)?
    thick = (zf - zb)[both]
    thick_cv = float(thick.std() / max(abs(thick.mean()), 1e-9))
    geom_ssim = ssim(np.where(both, zf, 0.0) / rng_z, np.where(both, -zb, 0.0) / rng_z)
    # colour: grey-level SSIM of the front and the flipped back
    gf = rf[..., :3].mean(-1)
    gb = rb_m[..., :3].mean(-1)
    textured = float(gf[both].std()) > 0.03 and colours is not None
    col_ssim = ssim(np.where(both, gf, 0.5), np.where(both, gb, 0.5)) if textured else float("nan")
    copy_geom = geom_ssim > cfg.mirror_geom_fail and thick_cv < 0.08
    copy_col = textured and col_ssim > cfg.mirror_ssim_fail
    fail = (copy_geom or copy_col) and not symmetric_ok
    s = 1.0
    if not symmetric_ok:
        s = min(1.0 - _soft(geom_ssim, cfg.mirror_geom_fail - 0.02, cfg.mirror_geom_fail) * (1.0 if thick_cv < 0.12 else 0.0),
                1.0 - (_soft(col_ssim, cfg.mirror_ssim_fail - 0.04, cfg.mirror_ssim_fail) if textured else 0.0))
    return {"score": float(s), "ok": not fail, "geom_ssim": geom_ssim, "colour_ssim": col_ssim, "thickness_cv": thick_cv,
            "copy_geometry": bool(copy_geom), "copy_colour": bool(copy_col), "textured": bool(textured)}


def gate_colour(colours, cfg: GateConfig) -> Dict:
    if colours is None:
        return {"score": None, "ok": True, "note": "no colours"}
    c = np.asarray(colours, np.float64)
    if not np.isfinite(c).all():
        return {"score": 0.0, "ok": False, "note": "NaN/inf colours"}
    black = float((c.max(1) < 0.035).mean())
    magenta = float(((c[:, 0] > 0.9) & (c[:, 1] < 0.12) & (c[:, 2] > 0.9)).mean())
    spread = float(c.std(0).mean())
    s = (1.0 - _soft(black, 0.05, 0.5)) * (1.0 - _soft(magenta, 0.0, 0.1)) * (0.6 + 0.4 * _soft(spread, 0.0, 0.05))
    return {"score": float(s), "ok": black < 0.5 and magenta < 0.1, "black_share": black, "magenta_share": magenta, "spread": spread}


def gate_fidelity(v, f, colours, targets: Sequence[TargetView], cfg: GateConfig) -> Dict:
    ious, cols, per = [], [], {}
    for t in targets:
        S = t.image.shape[0]
        img, _, _ = render(v, f, colours, t.az, t.el, S)
        m_t, m_g = t.image[..., 3] > 0.5, img[..., 3] > 0.5
        u = np.logical_or(m_t, m_g).sum()
        iou = float(np.logical_and(m_t, m_g).sum() / u) if u else 1.0
        both = m_t & m_g
        mae = float(np.abs(img[..., :3][both] - t.image[..., :3][both]).mean()) if both.sum() > 10 and colours is not None else float("nan")
        ious.append(iou)
        if np.isfinite(mae):
            cols.append(mae)
        per[t.name] = {"iou": iou, "colour_mae": mae}
    iou = float(np.mean(ious))
    col_s = 1.0 - _soft(float(np.mean(cols)), 0.04, 0.25) if cols else 1.0
    s = 0.75 * _soft(iou, 0.55, 0.95) + 0.25 * col_s
    return {"score": s, "ok": min(ious) >= cfg.min_view_iou, "mean_iou": iou, "min_iou": float(min(ious)),
            "colour_mae": float(np.mean(cols)) if cols else float("nan"), "per_view": per}


# ------------------------------------------------------------------------------------------------ the verdict
def verify(verts, faces, colours=None, targets: Optional[Sequence[TargetView]] = None, expect_solid: bool = True,
           symmetric_ok: bool = False, cfg: Optional[GateConfig] = None) -> Verdict:
    cfg = cfg or GateConfig()
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    if len(f) < 8 or not np.isfinite(v).all():
        return Verdict(False, 0.0, {"mesh": {"score": 0.0, "ok": False, "note": "empty or non-finite mesh"}}, ["mesh"])
    g: Dict[str, Dict] = {}
    g["mesh"] = gate_mesh(v, f, cfg, expect_solid)
    audit = g["mesh"].pop("_audit")
    g["depth"] = gate_depth(v, f, cfg, expect_solid, audit)
    g["back"] = gate_back(v, f, colours, cfg, symmetric_ok)
    g["colour"] = gate_colour(colours, cfg)
    if targets:
        g["fidelity"] = gate_fidelity(v, f, colours, targets, cfg)
    w = {k: wt for k, wt in cfg.weights.items() if k in g and g[k].get("score") is not None}
    tot = sum(w.values())
    score = sum(g[k]["score"] * wt for k, wt in w.items()) / max(tot, 1e-9)
    failing = [k for k, d in g.items() if not d.get("ok", True)]
    hard = [k for k in failing if k in ("mesh", "depth", "fidelity", "back")]
    if hard:
        score *= 0.5
    return Verdict(bool(not failing and score >= cfg.pass_score), float(score), g, failing)


def calibrate(items: Sequence[Dict], cfg: Optional[GateConfig] = None) -> Dict:
    """items: dicts with verts / faces / colours (assets you trust).  Prints nothing; returns the distribution of every
    gate number so you can set thresholds from evidence rather than from this file's defaults."""
    rows = []
    for it in items:
        d = verify(it["verts"], it["faces"], it.get("colours"), it.get("targets"), it.get("expect_solid", True),
                   it.get("symmetric_ok", False), cfg).as_dict()
        rows.append(d)
    out: Dict[str, Dict] = {}
    for gname in ("mesh", "depth", "back", "colour", "fidelity"):
        vals: Dict[str, List[float]] = {}
        for r in rows:
            for k, x in r["gates"].get(gname, {}).items():
                if isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x):
                    vals.setdefault(k, []).append(float(x))
        if vals:
            out[gname] = {k: {"p05": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}
                          for k, a in vals.items()}
    out["pass_rate"] = float(np.mean([r["passed"] for r in rows])) if rows else float("nan")
    out["mean_score"] = float(np.mean([r["score"] for r in rows])) if rows else float("nan")
    return out


# ------------------------------------------------------------------------------------------------ G6 file validity
def check_glb(path_or_bytes, max_bytes: int = 64 * 1024 * 1024) -> Dict:
    """Structural check of a GLB: header, JSON chunk, accessors inside the binary chunk, indices in range, finite
    positions.  (The Khronos glTF-Validator is the authority; this catches what a build step can break.)"""
    import json
    import struct
    from pathlib import Path

    data = Path(path_or_bytes).read_bytes() if isinstance(path_or_bytes, (str, Path)) else bytes(path_or_bytes)
    errs: List[str] = []
    if len(data) < 28 or data[:4] != b"glTF":
        return {"ok": False, "errors": ["not a GLB"], "bytes": len(data)}
    if struct.unpack("<I", data[4:8])[0] != 2:
        errs.append("glTF version is not 2")
    total = struct.unpack("<I", data[8:12])[0]
    if total != len(data):
        errs.append(f"header length {total} != file length {len(data)}")
    off, doc, binc = 12, None, b""
    while off + 8 <= len(data):
        ln, ct = struct.unpack("<II", data[off:off + 8])
        body = data[off + 8:off + 8 + ln]
        if ct == 0x4E4F534A:
            doc = json.loads(body.decode("utf8").rstrip(" \x00"))
        elif ct == 0x004E4942:
            binc = body
        off += 8 + ln + ((4 - ln % 4) % 4) * 0
        off = (off + 3) // 4 * 4
    if doc is None:
        return {"ok": False, "errors": errs + ["no JSON chunk"], "bytes": len(data)}
    views, accs = doc.get("bufferViews", []), doc.get("accessors", [])
    for i, a in enumerate(accs):
        bv = views[a["bufferView"]] if "bufferView" in a else None
        if bv is None:
            continue
        need = bv.get("byteOffset", 0) + bv["byteLength"]
        if need > len(binc):
            errs.append(f"accessor {i}: bufferView beyond the binary chunk")
    nverts = 0
    for m in doc.get("meshes", []):
        for p in m.get("primitives", []):
            pa = accs[p["attributes"]["POSITION"]]
            nverts = pa["count"]
            if "indices" in p:
                ia = accs[p["indices"]]
                bv = views[ia["bufferView"]]
                dt = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}[ia["componentType"]]
                idx = np.frombuffer(binc, dt, ia["count"], bv.get("byteOffset", 0) + ia.get("byteOffset", 0))
                if idx.size and int(idx.max()) >= nverts:
                    errs.append("index out of range")
                if idx.size % 3:
                    errs.append("index count not a multiple of 3")
            bv = views[pa["bufferView"]]
            pos = np.frombuffer(binc, np.float32, nverts * 3, bv.get("byteOffset", 0) + pa.get("byteOffset", 0))
            if not np.isfinite(pos).all():
                errs.append("non-finite positions")
    if len(data) > max_bytes:
        errs.append(f"file is {len(data) / 1e6:.1f} MB, budget {max_bytes / 1e6:.0f} MB")
    return {"ok": not errs, "errors": errs, "bytes": len(data), "vertices": nverts}
