"""`aura-white doctor3d`: what this machine can do, which generators are ready, and a self-test of the engine's own logic.

The self-test builds a synthetic object, renders pictures of it, damages a copy and repairs it with the pictures - so a
broken install (or a numpy/scipy problem) shows up here in seconds, before a long generation does.  `--deep` additionally
loads the shape model and generates a small mesh, which is the only way to confirm the neural stack end to end.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from .hardware import STAGES, Hardware, plan_stage, probe_hardware


def _ok(flag: bool) -> str:
    return "ok" if flag else "FAIL"


def _selftest() -> List[tuple]:
    from . import meshutil as M
    from .audit import audit_mesh
    from .fusion import build_constraints, fuse
    from .views import ViewCam, ViewSpec, render_maps

    out = []
    # a sphere with a bump, and a flat plate
    th, ph = np.meshgrid(np.linspace(0, np.pi, 34)[1:-1], np.linspace(0, 2 * np.pi, 64, endpoint=False), indexing="ij")
    ring = np.stack([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)], -1).reshape(-1, 3)
    v = np.vstack([[0, 1, 0], ring, [0, -1, 0]])
    nl, nlon = 32, 64
    idx = lambda i, j: 1 + i * nlon + (j % nlon)
    f = [[0, idx(0, j + 1), idx(0, j)] for j in range(nlon)]
    for i in range(nl - 2):
        for j in range(nlon):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            f += [[a, b, c], [a, c, d]]
    f += [[len(v) - 1, idx(nl - 2, j), idx(nl - 2, j + 1)] for j in range(nlon)]
    f = np.asarray(f)
    t = time.time()
    lo, side = M.grid_for(v)
    occ = M.voxelize(v, f, 64, lo, side)
    vol = occ.sum() * (side / 64) ** 3
    out.append(("voxelisation", abs(vol - 4 / 3 * np.pi) / (4 / 3 * np.pi) < 0.05, f"sphere volume {vol:.3f} vs {4 / 3 * np.pi:.3f}"))
    a = audit_mesh(v, f)
    out.append(("3-D audit (solid)", a.flat_risk < 0.05 and a.profile == "solid", f"depth {a.depth_ratio:.2f}"))
    slab_v = v * np.array([1.0, 1.0, 0.08])
    a2 = audit_mesh(slab_v, f)
    out.append(("3-D audit (flat detection)", a2.flat_risk > 0.9, f"flat-risk {a2.flat_risk:.2f}"))
    views = [(ViewSpec.parse(n), render_maps(v, f, ViewCam(ViewSpec.parse(n).az, 0.0, 192, 80.0, 0.0, 0.0))[0]) for n in ("front", "left", "back")]
    cons = build_constraints(v * 0.8, f, views)
    r = fuse(v * 0.8, f, cons, grid=64)
    out.append(("fusion against pictures", r.stats.get("iou_after", r.stats["iou_before"]) >= r.stats["iou_before"] - 1e-6,
                f"agreement {r.stats['iou_before']:.3f} -> {r.stats.get('iou_after', r.stats['iou_before']):.3f}"))
    out.append(("self-test time", True, f"{time.time() - t:.1f}s"))
    return out


def run_doctor(device: str = "auto", deep: bool = False) -> int:
    hw = probe_hardware(device)
    print(f"machine: {hw.describe()}")
    print("\nmemory plan per stage (quality is identical in every row; only the way it is executed differs):")
    for name in STAGES:
        p = plan_stage(name, hw)
        flag = "" if p.feasible else "   <-- cannot run here"
        print(f"  {name:22s} {p.device:5s} {p.offload:10s} {p.dtype:9s} chunks x{p.chunk_scale:<6g} ~{p.est_peak_gb:4.1f} GB{flag}")
    print("\nwhat the same recipe needs at other sizes (for comparison):")
    for gb in (4, 6, 8, 12, 24):
        h = Hardware("cuda", f"{gb} GB card", gb, gb, max(hw.ram_total_gb, 16), max(hw.ram_free_gb, 12), True, True)
        p = plan_stage("hunyuan3d2_shape", h)
        print(f"  {gb:>2} GB GPU: shape model -> {p.offload:10s} (peak ~{p.est_peak_gb:.1f} GB)")
    print("\nshape generators:")
    from .shapegen import Hunyuan3D2Shape
    from .intake import ViewInput  # noqa: F401
    for b in (Hunyuan3D2Shape(False), Hunyuan3D2Shape(True)):
        pr = b.probe(hw, [])
        print(f"  {b.name:14s} {'READY' if pr.ok else 'not ready'}   {pr.reason}")
    from .textgen import TextToImage
    ok, why, _ = TextToImage().probe(hw)
    print(f"  {'text -> picture':14s} {'READY' if ok else 'not ready'}   {why}")
    print("\nself-test of the engine's own logic:")
    bad = 0
    for name, ok, detail in _selftest():
        print(f"  [{_ok(ok):4s}] {name}: {detail}")
        bad += 0 if ok else 1
    if deep:
        print("\ndeep test: loading the shape model and sculpting a small mesh ...")
        try:
            from PIL import Image, ImageDraw
            from .controller import GenCall
            from .intake import request_from_arrays
            img = Image.new("RGB", (384, 384), (240, 240, 240))
            ImageDraw.Draw(img).ellipse((96, 60, 288, 324), fill=(200, 80, 60))
            m = np.asarray(Image.new("L", (384, 384), 0))
            mm = Image.new("L", (384, 384), 0)
            ImageDraw.Draw(mm).ellipse((96, 60, 288, 324), fill=255)
            req = request_from_arrays(np.asarray(img), np.asarray(mm) / 255.0)
            b = Hunyuan3D2Shape(False)
            pr = b.probe(hw, req.views)
            if not pr.ok:
                print("  cannot run:", pr.reason)
                return 1
            t = time.time()
            v, f = b.generate(GenCall(0, 128, 10, 5.0), req.views, pr.plan, print)
            b.close()
            print(f"  [ ok ] generated {len(f):,} faces in {time.time() - t:.0f}s")
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")
            bad += 1
    return 1 if bad else 0
