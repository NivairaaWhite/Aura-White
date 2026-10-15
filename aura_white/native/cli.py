"""`aura-white native ...` and `aura-white sheet ...`: the command line of Aura's own-model track.

  sheet     front / back / left / right pictures (+ optionally a mesh from any generator) -> a textured GLB   (no GPU, no weights)
  native    doctor | licences | budget | make-data | ingest | teach | train-vae | train-flow | evolve | distill | prune |
            bench | approve | sample                                                              (needs PyTorch; a GPU for real sizes)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _say(msg: str):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------ sheet
def cmd_sheet(a) -> int:
    from . import multiview as MV

    src = {k: getattr(a, k) for k in ("front", "back", "left", "right") if getattr(a, k)}
    for extra in a.view or []:
        name, _, path = extra.partition("=")
        if not path:
            raise SystemExit(f"--view expects name=path, got {extra!r}")
        src[name] = path
    if not src:
        raise SystemExit("give at least --front (and --left or --right for the depth); see aura-white sheet --help")
    for k, p in src.items():
        if not Path(p).is_file():
            raise SystemExit(f"{k}: {p} not found")
    try:
        res = MV.reconstruct(src, out_dir=a.out, name=a.name, height_m=a.height, faces=a.faces, cells=a.cells, roundness=a.roundness,
                             tex=a.tex, canvas=a.canvas, unlit=a.unlit, wing_p=a.wing_p, base_mesh=a.mesh, say=_say)
    except ValueError as e:                                   # bad input (empty picture, unknown view): a message, not a traceback
        raise SystemExit(f"sheet: {e}")
    r = res.report
    print(f"\n{r['mode']}: {r['faces']:,} faces, {r['size_m']} m")
    print("outline agreement with your pictures: " + ", ".join(f"{k} {v:.3f}" for k, v in r["outline_iou"].items()))
    for n in r["notes"]:
        print("  note: " + n)
    print(f"wrote {res.glb}  (+ albedo, turntable preview and report in {a.out})")
    return 0


# ------------------------------------------------------------------------------------------------ native
def _need_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise SystemExit(f"PyTorch is required for this command ({e}); install it, e.g. pip install torch")


def cmd_doctor(a) -> int:
    _need_torch()
    import torch

    from .model import AuraNative
    from .procedural import make_asset
    from .gates import verify

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  threads={torch.get_num_threads()}")
    m = AuraNative.new("tiny")
    print("tiny preset parameters:", m.params())
    t0 = time.time()
    asset = make_asset("figure", 1)
    v, f, c = asset.mesh(48)
    ver = verify(v, f, c, None, True)
    print(f"gates on a procedural figure: passed={ver.passed} score={ver.score:.2f} failing={ver.failing}")
    print(f"self-test {time.time() - t0:.1f}s - ok")
    return 0


def cmd_licences(a) -> int:
    from .licenses import ledger_table

    for r in ledger_table():
        print(f"{r['status']:<14}{r['key']:<14}{r['name']}\n{'':<28}{r['why']}")
    return 0


def cmd_budget(a) -> int:
    _need_torch()
    from .compress import budget
    from .config import preset

    for name in a.preset or ["tiny", "s", "m", "l"]:
        b = budget(preset(name), steps=a.steps, batch=a.batch, gpu_tflops=a.gpu_tflops, utilisation=a.utilisation)
        print(json.dumps(b, indent=1))
    return 0


def cmd_make_data(a) -> int:
    from .data import DataSpec, build_procedural

    spec = DataSpec(n_points=a.points, n_queries=a.queries, image_size=a.image_size)
    dd = build_procedural(a.out, a.n, spec, seed0=a.seed0, workers=a.workers)
    print(f"{len(dd)} procedural assets in {a.out}")
    return 0


def cmd_ingest(a) -> int:
    from .data import DataSpec, ingest_folder

    lic = json.loads(Path(a.licences).read_text()) if a.licences else None
    dd = ingest_folder(a.folder, a.out, DataSpec(image_size=a.image_size), lic, commercial=not a.research,
                       default_licence=a.default_licence, limit=a.limit)
    print(f"{len(dd)} assets in {a.out}; commercially clean: {dd.prov.commercially_clean()[0]}")
    return 0


def cmd_teach(a) -> int:
    from .data import DataSpec
    from .teach import CommandTeacher, teach

    t = CommandTeacher(a.teacher, a.command, timeout=a.timeout, commercial=not a.research, accept_risk=a.accept_risk)
    dd = teach(t, a.pictures, a.out, DataSpec(image_size=a.image_size), limit=a.limit)
    print(json.dumps(dd.report, indent=1))
    return 0


def _load_or_new(a):
    from .model import AuraNative

    if a.ckpt and Path(a.ckpt).exists():
        return AuraNative.load(a.ckpt, "cpu")
    return AuraNative.new(a.preset)


def cmd_train_vae(a) -> int:
    _need_torch()
    from .data import ShardDataset
    from .train import train_vae

    m = _load_or_new(a)
    ds = ShardDataset(a.data, "train")
    start = int(m.card.get("vae_steps", 0)) if a.resume else 0
    r = train_vae(m, ds, steps=a.steps, batch=a.batch, lr=a.lr, device=a.device, ckpt=a.ckpt, start_step=start,
                  stop_after=a.minutes * 60 if a.minutes else None)
    print(f"vae: step {r['next_step']}/{a.steps} done={r['done']}  -> {a.ckpt}")
    return 0


def cmd_train_flow(a) -> int:
    _need_torch()
    from .data import ShardDataset
    from .train import fit_latent_stats, train_flow

    m = _load_or_new(a)
    ds = ShardDataset(a.data, "train")
    start = int(m.card.get("flow_steps", 0)) if a.resume else 0
    if start == 0:
        fit_latent_stats(m, ds)
    r = train_flow(m, ds, steps=a.steps, batch=a.batch, lr=a.lr, device=a.device, ckpt=a.ckpt, start_step=start,
                   stop_after=a.minutes * 60 if a.minutes else None)
    print(f"flow: step {r['next_step']}/{a.steps} done={r['done']}  -> {a.ckpt}")
    return 0


def _bench_items(a, split="val"):
    from .bench import items_from_procedural
    from .data import ShardDataset

    return items_from_procedural(ShardDataset(a.data, split), a.n)


def cmd_evolve(a) -> int:
    _need_torch()
    from .bench import evaluate
    from .data import ShardDataset
    from .evolve import EvolveConfig, evolve
    from .model import AuraNative

    champ = AuraNative.load(a.ckpt, "cpu")
    anchor = ShardDataset(a.data, "train")
    pool = ShardDataset(a.pool, "all")
    items = _bench_items(a)
    cfg = EvolveConfig(rounds=a.rounds, prompts=a.prompts, k=a.k, finetune_steps=a.finetune_steps, selection=a.selection,
                       anchor_frac=a.anchor_frac)
    res = evolve(champ, anchor, pool, items, cfg, a.out, say=_say)
    res["champion"].save(Path(a.out) / "champion.pt")
    for h in res["history"]:
        print(json.dumps({k: h[k] for k in h if k in ("round", "promoted", "reason")}))
    return 0


def cmd_distill(a) -> int:
    _need_torch()
    from .compress import distill_guidance
    from .data import ShardDataset
    from .model import AuraNative

    m = AuraNative.load(a.ckpt, "cpu")
    r = distill_guidance(m, ShardDataset(a.data, "train"), steps=a.steps, batch=a.batch, device=a.device)
    r["model"].save(a.out)
    print(f"guidance distilled -> {a.out}  {r['report']}")
    return 0


def cmd_prune(a) -> int:
    _need_torch()
    from .compress import prune_depth
    from .data import ShardDataset
    from .model import AuraNative

    m = AuraNative.load(a.ckpt, "cpu")
    r = prune_depth(m, ShardDataset(a.data, "train"), keep=a.keep, heal_steps=a.steps, batch=a.batch, device=a.device)
    r["model"].save(a.out)
    print(f"pruned -> {a.out}  kept blocks {r['kept']}  {r['report']}")
    return 0


def cmd_bench(a) -> int:
    _need_torch()
    from .compress import measure, quantize_int8
    from .model import AuraNative

    m = AuraNative.load(a.ckpt, "cpu")
    if a.int8:
        m = quantize_int8(m)
    r = measure(m, _bench_items(a), seeds=tuple(range(a.seeds)), steps=a.steps)
    print(json.dumps(r, indent=1, default=float))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1, default=float))
    return 0


def cmd_approve(a) -> int:
    """The only way a checkpoint becomes eligible as a default back-end.  Deliberately hard to pass with demo data."""
    _need_torch()
    from .data import DataDir
    from .model import AuraNative

    m = AuraNative.load(a.ckpt, "cpu")
    rep = json.loads(Path(a.bench).read_text())
    s = rep.get("summary", rep)
    real, total, clean, why_dirty = 0, 0, True, []
    for d in a.data:
        dd = DataDir(d)
        total += len(dd)
        real += sum(1 for it in dd.items if "family" not in it)             # procedural records carry a 'family'; real ones do not
        ok, why = dd.prov.commercially_clean()
        if not ok:
            clean = False
            why_dirty += why
    problems = []
    if real < a.min_real:
        problems.append(f"only {real} real (non-procedural) training assets across {total}; need >= {a.min_real} - a model trained "
                        "on procedural shapes has not seen the world and must not be a default")
    if s.get("pass_rate", 0.0) < a.min_pass:
        problems.append(f"benchmark pass rate {s.get('pass_rate', 0.0):.2f} < {a.min_pass}")
    if s.get("score", 0.0) < a.min_score:
        problems.append(f"benchmark verifier score {s.get('score', 0.0):.2f} < {a.min_score}")
    if not a.research and not clean:
        problems.append("training data is not commercially clean: " + "; ".join(why_dirty))
    if problems:
        print("NOT approved:\n  - " + "\n  - ".join(problems))
        return 1
    m.save(a.ckpt, {"approved_for_default": True, "approval": {"real_assets": real, "pass_rate": s.get("pass_rate"),
                                                              "score": s.get("score"), "bench": str(a.bench),
                                                              "commercially_clean": clean, "time": time.strftime("%Y-%m-%d %H:%M:%S")}})
    print(f"approved: {a.ckpt}")
    return 0


def cmd_sample(a) -> int:
    _need_torch()
    import numpy as np
    from PIL import Image

    from ..mesh.export import save_mesh
    from .model import AuraNative, ViewIn
    from .render import prep_picture

    m = AuraNative.load(a.ckpt, "cpu")
    size = m.cfg.flow.image_size
    views = []
    for spec in a.picture:
        name, _, path = spec.rpartition("=")
        az = {"": 0.0, "front": 0.0, "back": 180.0, "left": 90.0, "right": -90.0}.get(name, None)
        if az is None:
            az = float(name)
        im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
        views.append(ViewIn(prep_picture(im, size), az, 0.0))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(m.generate(views, n=a.n, steps=a.steps, cfg_scale=a.cfg, seed=a.seed, res=a.res)):
        p = out / f"sample_{i}.glb"
        save_mesh(p, g.verts, g.faces, (np.clip(g.colours, 0, 1) * 255).astype(np.uint8))
        print(f"{p}  {len(g.faces):,} faces")
    print("note: " + ("this checkpoint's model card says it is NOT approved as a default (demo / unverified)"
                      if not m.card.get("approved_for_default") else "checkpoint is approved"))
    return 0


# ------------------------------------------------------------------------------------------------ parsers
def add_parsers(sub: "argparse._SubParsersAction"):
    sh = sub.add_parser("sheet", help="front/back/left/right pictures -> textured GLB (carved from the outlines, or painting a mesh you already have)")
    for k in ("front", "back", "left", "right"):
        sh.add_argument(f"--{k}", help=f"the {k} picture (cut-out PNG, or any picture on a plain background)")
    sh.add_argument("--view", action="append", help="extra view as name=path (used for painting only)")
    sh.add_argument("--mesh", help="keep THIS mesh (from TRELLIS / Hunyuan / TripoSR / Meshy / your sculpt) and only paint it from the pictures")
    sh.add_argument("--out", default="sheet_out")
    sh.add_argument("--name", default="model")
    sh.add_argument("--height", type=float, default=1.0, help="object height in metres (the character in the notes is 1.68)")
    sh.add_argument("--faces", type=int, default=60000)
    sh.add_argument("--cells", type=int, default=420, help="carving resolution (voxels along the height)")
    sh.add_argument("--roundness", type=float, default=2.4, help="cross-section exponent: 2 = ellipse, larger = boxier")
    sh.add_argument("--wing-p", type=float, default=None, help="pinch very wide, shallow sections (sleeves) to this exponent")
    sh.add_argument("--tex", type=int, default=2048)
    sh.add_argument("--canvas", type=int, default=1536)
    sh.add_argument("--unlit", action="store_true", help="flat (unlit) material, for a cel-shaded look")
    sh.set_defaults(fn=cmd_sheet)

    n = sub.add_parser("native", help="Aura's own model: data, teachers, training, self-improvement, compression, benchmark")
    ns = n.add_subparsers(dest="native_cmd", required=True)

    def add(name, fn, help):
        p = ns.add_parser(name, help=help)
        p.set_defaults(fn=fn)
        return p

    add("doctor", cmd_doctor, "check PyTorch and run a 10-second self-test")
    add("licences", cmd_licences, "print the licence ledger: who may teach Aura, who may not, and why")
    p = add("budget", cmd_budget, "compute: parameters, FLOPs and GPU-hours per preset (an estimate, labelled)")
    p.add_argument("--preset", nargs="*")
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--gpu-tflops", type=float, default=120.0)
    p.add_argument("--utilisation", type=float, default=0.35)
    p = add("make-data", cmd_make_data, "build a dataset of procedural assets with exact ground truth")
    p.add_argument("out")
    p.add_argument("-n", type=int, default=600)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--points", type=int, default=1024)
    p.add_argument("--queries", type=int, default=2048)
    p.add_argument("--image-size", type=int, default=64)
    p = add("ingest", cmd_ingest, "add a folder of meshes (licence-filtered) to a dataset")
    p.add_argument("folder")
    p.add_argument("out")
    p.add_argument("--licences", help="JSON {file stem: licence tag}")
    p.add_argument("--default-licence", default="own")
    p.add_argument("--research", action="store_true", help="non-commercial research mode (accepts by-nc / by-sa)")
    p.add_argument("--limit", type=int)
    p.add_argument("--image-size", type=int, default=64)
    p = add("teach", cmd_teach, "ask a licensed teacher model to make meshes from pictures; keep what passes the gates")
    p.add_argument("pictures", help="folder of cut-out PNGs")
    p.add_argument("out")
    p.add_argument("--teacher", required=True, help="ledger key, e.g. trellis2 | triposg | triposr  (hunyuan3d is refused)")
    p.add_argument("--command", required=True, help='your command, e.g. "python run_trellis2.py --image {image} --out {out}"')
    p.add_argument("--accept-risk", action="store_true", help="use a teacher whose licence is unverified (recorded in the manifest)")
    p.add_argument("--research", action="store_true")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--limit", type=int)
    p.add_argument("--image-size", type=int, default=64)
    for name, fn, h in (("train-vae", cmd_train_vae, "train the shape VAE"), ("train-flow", cmd_train_flow, "train the generator")):
        p = add(name, fn, h)
        p.add_argument("data")
        p.add_argument("--ckpt", required=True)
        p.add_argument("--preset", default="tiny", choices=["tiny", "s", "m", "l"])
        p.add_argument("--steps", type=int, default=2000)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--device", default="auto")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--minutes", type=float, help="stop (and save) after this long; run again with --resume")
    p = add("evolve", cmd_evolve, "verifier-gated self-improvement rounds (promotes only on a better frozen benchmark)")
    p.add_argument("data")
    p.add_argument("pool", help="dataset whose pictures are used as prompts")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="evolve_out")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--prompts", type=int, default=120)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--finetune-steps", type=int, default=600)
    p.add_argument("--selection", default="verifier", choices=["verifier", "random"])
    p.add_argument("--anchor-frac", type=float, default=0.5)
    p.add_argument("-n", type=int, default=60, help="benchmark items")
    p = add("distill", cmd_distill, "bake classifier-free guidance into the weights (about 2x faster sampling)")
    p.add_argument("ckpt")
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p = add("prune", cmd_prune, "drop the least important generator blocks, then heal by distillation")
    p.add_argument("ckpt")
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--keep", type=int, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p = add("bench", cmd_bench, "frozen benchmark: oracle + verifier metrics, size and speed")
    p.add_argument("ckpt")
    p.add_argument("data")
    p.add_argument("-n", type=int, default=60)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--out")
    p = add("approve", cmd_approve, "mark a checkpoint eligible as a default back-end - only after real data and a passing benchmark")
    p.add_argument("ckpt")
    p.add_argument("--bench", required=True, help="the JSON written by `native bench --out`")
    p.add_argument("--data", nargs="+", required=True, help="the dataset folders it was trained on (provenance is read from them)")
    p.add_argument("--min-real", type=int, default=5000)
    p.add_argument("--min-pass", type=float, default=0.90)
    p.add_argument("--min-score", type=float, default=0.85)
    p.add_argument("--research", action="store_true", help="allow non-commercially-clean data (the checkpoint then carries that restriction)")
    p = add("sample", cmd_sample, "pictures -> meshes with a checkpoint")
    p.add_argument("ckpt")
    p.add_argument("picture", nargs="+", help="path or view=path (front|back|left|right|<azimuth>=path); first = the frame")
    p.add_argument("--out", default="native_samples")
    p.add_argument("-n", type=int, default=4)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--res", type=int, default=64)
