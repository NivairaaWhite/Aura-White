"""Aura 3D engine: text / one picture / several pictures  ->  a real, closed, textured 3-D asset.

    from aura_white.gen3d import Aura3D, MakeOptions
    Aura3D(MakeOptions()).make(["knight_front.png", "knight_back.png"], "out")
    Aura3D().make(["a bronze dragon statue"], "out")                       # text
    Aura3D().refine("out/knight", add=["left=side.png"])                   # more pictures, new version

Pipeline:  outline every picture  ->  (text: draw + rank reference pictures)  ->  choose the shape generator the machine can
run at full recipe quality  ->  quality loop (generate, orient, fit to pictures, fuse, audit, re-roll)  ->  paint the sides
and back  ->  asset pack (LODs, UVs, normal/ORM maps, collision) with every picture projected into the atlas.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .controller import LoopResult, run_quality_loop
from .hardware import Hardware, probe_hardware
from .intake import Request, ViewInput, load_request
from .job import Job, slug
from .paint import ExtraView
from .recipe import Recipe
from .shapegen import Choice, NoBackend, choose_backend
from .views import ViewSpec

Say = Callable[[str], None]


@dataclass
class MakeOptions:
    recipe: Recipe = field(default_factory=Recipe)
    device: str = "auto"
    tier: str = "universal"                 # universal | max
    allow_basic: bool = False               # accept the built-in network / visual hull when no neural back-end can run
    backend: Optional[str] = None           # force one back-end by name
    seed: int = 0
    kind: str = "auto"                      # asset kind for the pack (auto | character | prop | building | ...)
    name: Optional[str] = None
    pack: bool = True
    style: Optional[str] = None             # extra words for the text-to-image prompt
    t2i_model: Optional[str] = None
    paint: Optional[bool] = None            # None: follow the recipe
    pick: Optional[int] = None              # use text-to-image candidate N instead of the best-ranked
    outline_kind: str = "auto"              # transparent | plain | lineart | illustration | photo | auto
    asset: Dict = field(default_factory=dict)   # extra AssetOptions fields (height, lods, ...)


@dataclass
class Result:
    out_dir: Path
    version: str
    mesh: Path
    pack: Optional[Path]
    passed: bool
    reason: str
    summary: Dict
    notes: List[str] = field(default_factory=list)


def _cutout_rgb(v: ViewInput) -> np.ndarray:
    a = np.asarray(v.image.convert("RGBA"))
    return a[..., :3].copy()


class Aura3D:
    def __init__(self, options: Optional[MakeOptions] = None, log: Optional[Say] = None):
        self.o = options or MakeOptions()
        self.log = log or print

    # ------------------------------------------------------------------------------------------ public
    def make(self, items: Sequence[str], out_dir, text: Optional[str] = None) -> Result:
        o, say = self.o, self.log
        hw = probe_hardware(o.device)
        say(f"[1/6] machine: {hw.describe()}")
        say(f"      recipe '{o.recipe.name}' (quality signature {o.recipe.signature()}) - identical on every machine; only speed differs")
        from .intake import split_inputs
        files, words = split_inputs(items)
        prompt = " ".join(([text] if text else []) + words).strip() or None
        name = o.name or slug(prompt or (files[0][1].stem if files else "asset"))
        root = Path(out_dir) / name
        job = Job.create(root, name=name, items=[str(i) for i in items], text=prompt, recipe=self._recipe_dict(), options=self._opt_dict(),
                         hardware=hw.describe())
        request = self._reference(items, text, job, hw)
        return self._build(job, request, hw)

    def refine(self, job_dir, add: Sequence[str] = (), pick: Optional[int] = None, seed: Optional[int] = None,
               feedback: Optional[str] = None, recipe: Optional[Recipe] = None) -> Result:
        """A new version of an existing job: more pictures, another text-to-image candidate, other seeds, a different recipe."""
        job = Job.load(job_dir)
        o, say = self.o, self.log
        hw = probe_hardware(o.device)
        if recipe is not None:
            o.recipe = recipe
        o.seed = seed if seed is not None else (o.seed + 7919 * (len(job.versions) + 1))
        if feedback:
            job.data.setdefault("feedback", []).append(feedback)
            say(f"feedback noted: {feedback}")
        inputs = sorted(str(p) for p in (job.root / "inputs").glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        if pick is not None:
            cands = sorted((job.root / "views" / "candidates").glob("candidate_*.png"))
            if not cands or not (1 <= pick <= len(cands)):
                raise ValueError(f"--pick must be 1..{len(cands)} (candidates are in {job.root / 'views' / 'candidates'})")
            ref = job.root / "inputs" / "front.png"
            Image.open(cands[pick - 1]).save(ref)
            inputs = [str(p) for p in inputs if Path(p).name != "front.png"] + [str(ref)]
        # a picture you add REPLACES the stored one for the same view (e.g. a better back photo)
        from .views import canonical_label, infer_label
        replaced = set()
        for a in add:
            lab = a.partition("=")[0] if "=" in a and not Path(a).exists() else infer_label(a)
            lab = canonical_label(lab.partition("@")[0]) if lab else None
            if lab:
                replaced.add(lab)
        inputs = [p for p in inputs if canonical_label(Path(p).stem) not in replaced]
        labelled = []
        for p in inputs:
            stem = Path(p).stem
            labelled.append(f"{stem}={p}" if ViewSpecSafe.ok(stem) else p)
        for a in add:
            labelled.append(a)
        request = load_request(labelled, text=job.data.get("text"), kind=o.outline_kind, strict=True, say=say)
        if pick is not None:
            request = self._normalise_after_pick(request)
        self._save_inputs(job, request)
        return self._build(job, request, hw)

    # ------------------------------------------------------------------------------------------ steps
    def _recipe_dict(self) -> Dict:
        from dataclasses import asdict
        d = asdict(self.o.recipe)
        d["signature"] = self.o.recipe.signature()
        return d

    def _opt_dict(self) -> Dict:
        o = self.o
        return {"seed": o.seed, "tier": o.tier, "allow_basic": o.allow_basic, "backend": o.backend, "kind": o.kind,
                "t2i_model": o.t2i_model, "style": o.style, "outline_kind": o.outline_kind}

    def _normalise_after_pick(self, request: Request) -> Request:
        return request

    def _reference(self, items, text, job: Job, hw: Hardware) -> Request:
        o, say = self.o, self.log
        say("[2/6] reading the pictures and finding their outlines" if any(Path(str(i)).is_file() for i in items) else
            "[2/6] drawing reference pictures from the text")
        from .intake import split_inputs
        files, words = split_inputs(items)
        if files:
            req = load_request(items, text=text, kind=o.outline_kind, strict=True, say=say)
        else:
            prompt = " ".join(([text] if text else []) + words).strip()
            req = self._text_reference(prompt, job, hw)
        self._save_inputs(job, req)
        for n in req.notes:
            say("      " + n)
        return req

    def _text_reference(self, prompt: str, job: Job, hw: Hardware) -> Request:
        from .textgen import TextToImage, rank_references
        o, say = self.o, self.log
        t2i = TextToImage(o.t2i_model)
        ok, why, plan = t2i.probe(hw)
        if not ok:
            raise NoBackend("a text prompt needs a text-to-image model, which cannot run here: " + why +
                            "\nAlternatively give a picture: aura-white make picture.png")
        imgs = t2i.generate(prompt, o.recipe.t2i_candidates, o.recipe.t2i_size, o.seed, plan, say, o.style)
        t2i.close()
        cdir = job.root / "views" / "candidates"
        cdir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(imgs, 1):
            im.save(cdir / f"candidate_{i}.png")
        ranked = rank_references(imgs)
        say("      candidates ranked: " + ", ".join(f"#{i + 1}={s:.2f}" for i, s, _ in ranked))
        use = (o.pick - 1) if o.pick else ranked[0][0]
        ref = job.root / "inputs" / "front.png"
        imgs[use].save(ref)
        job.data["t2i"] = {"model": t2i.model, "chosen": use + 1, "ranking": [(i + 1, round(s, 3)) for i, s, _ in ranked]}
        job.save()
        return load_request([f"front={ref}"], text=prompt, kind=o.outline_kind, strict=True, say=say)

    def _save_inputs(self, job: Job, req: Request):
        for v in req.views:
            v.image.save(job.root / "inputs" / f"{v.spec.label}.png")
            prev = Image.new("RGB", v.image.size, (128, 128, 128))
            prev.paste(v.image, (0, 0), v.image)
            edge = np.asarray(prev).copy()
            from scipy import ndimage as ndi
            ring = v.mask & ~ndi.binary_erosion(v.mask, iterations=2)
            edge[ring] = (255, 40, 120)
            Image.fromarray(edge).save(job.root / "views" / f"outline_{v.spec.label}.png")

    def _build(self, job: Job, req: Request, hw: Hardware) -> Result:
        o, say = self.o, self.log
        recipe = o.recipe
        ref = req.reference()
        vid = job.next_version()
        vdir = job.root / vid
        t0 = time.time()
        best, loop, choice = self._shape(req, hw)
        mesh_path = vdir / "mesh.glb"
        from ..backends.glb import write_glb_mesh
        write_glb_mesh(mesh_path, best.verts, best.faces)
        say(f"      {'PASSED' if loop.passed else 'best effort'}: {loop.reason}")
        extra = self._extra_views(req, best.verts, best.faces, hw, job)
        pack_dir = None
        report: Dict = {}
        if o.pack:
            pack_dir = vdir / "pack"
            say("[5/6] building the asset pack (LODs, UVs, maps, collision) and projecting every view into the texture")
            report = self._pack(job, req, best, extra, pack_dir)
        entry = {"id": vid, "seconds": round(time.time() - t0, 1), "backend": choice.backend.name, "tier": choice.backend.tier,
                 "passed": loop.passed, "reason": loop.reason, "seed": o.seed, "best": best.summary(), "history": loop.history,
                 "mesh": str(mesh_path.relative_to(job.root)), "pack": str(pack_dir.relative_to(job.root)) if pack_dir else None,
                 "extra_views": [{"label": e.label, "source": e.source, "az": e.az} for e in extra],
                 "recipe_signature": recipe.signature(), "notes": req.notes + choice.notes + best.notes}
        (vdir / "report.json").write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")
        job.add_version(entry)
        self._write_report(job, entry, hw)
        say(f"[6/6] done in {entry['seconds']}s -> {pack_dir or mesh_path}")
        return Result(job.root, vid, mesh_path, pack_dir, loop.passed, loop.reason, entry, entry["notes"])

    def _shape(self, req: Request, hw: Hardware):
        """Choose the generator the machine can run at full recipe quality, then run the quality loop."""
        o, say = self.o, self.log
        recipe = o.recipe
        say(f"[3/6] choosing a shape generator ({len(req.views)} picture(s): {', '.join(v.spec.label for v in req.views)})")
        choice: Choice = choose_backend([v for v in req.views], hw, o.tier, o.allow_basic, say, o.backend)
        for n in choice.notes:
            say("      " + n)
        say(f"      -> {choice.backend.name} [{choice.backend.tier}] on {choice.plan.device}"
            f"{'' if choice.plan.offload in ('none', '') else ' with ' + choice.plan.offload + ' offload'}")
        masks = [(v.spec, v.mask) for v in req.views]
        say(f"[4/6] building the shape: up to {recipe.max_rounds} round(s) x {recipe.candidates} candidate(s)")
        try:
            loop: LoopResult = run_quality_loop(lambda call: choice.backend.generate(call, req.views, choice.plan, say),
                                                masks, recipe, base_seed=o.seed, say=say)
        finally:
            choice.backend.close()
        return loop.best, loop, choice

    def _extra_views(self, req: Request, v: np.ndarray, f: np.ndarray, hw: Hardware, job: Optional[Job] = None) -> List[ExtraView]:
        o, say = self.o, self.log
        out: List[ExtraView] = []
        ref = req.reference()
        for vi in req.views:
            if vi is ref:
                continue
            out.append(ExtraView(vi.spec.label, vi.spec.az, vi.spec.el, _cutout_rgb(vi), vi.mask.copy(), 1.0, "user"))
        use_paint = o.recipe.paint_views if o.paint is None else o.paint
        if not use_paint:
            return out
        from .viewsynth import DepthPainter, missing_azimuths
        want = missing_azimuths(req.views, o.recipe.paint_azimuths)
        if not want:
            return out
        painter = DepthPainter()
        ok, why, plan = painter.probe(hw)
        if not ok:
            say(f"      the unseen sides keep a smooth colour fill - the view painter cannot run here ({why})")
            return out
        try:
            say(f"      painting {len(want)} unseen view(s) from the mesh's depth")
            out += painter.paint(v, f, ref.image, want, ref.spec.el, req.text or "", plan, say, size=1024, seed=o.seed)
        except Exception as e:                                         # the asset is still valid without painted views
            say(f"      view painter failed ({type(e).__name__}: {e}) - the unseen sides keep a smooth colour fill")
        finally:
            painter.close()
        return out

    def _pack(self, job: Job, req: Request, best, extra: List[ExtraView], pack_dir: Path) -> Dict:
        from ..asset.pipeline import AssetOptions, build_asset
        o = self.o
        ref_png = job.root / "inputs" / f"{req.reference().spec.label}.png"
        opts = AssetOptions(kind=o.kind, name=job.data.get("name"), proxy=(best.verts, best.faces), proxy_yaw=0.0,
                            extra_views=extra or None, **o.asset)
        return build_asset(str(ref_png), str(pack_dir), opts, log=lambda s: self.log("      " + s))

    def _write_report(self, job: Job, e: Dict, hw: Hardware):
        b = e["best"]
        lines = [f"# {job.data.get('name')} - {e['id']}", "",
                 f"- result: **{'passed every check' if e['passed'] else 'best effort'}** - {e['reason']}",
                 f"- generator: {e['backend']} (tier: {e['tier']}), seed {e['seed']}, recipe signature `{e['recipe_signature']}`",
                 f"- machine: {hw.describe()}", f"- time: {e['seconds']} s", "",
                 "## Is it really 3-D?", f"- depth / width: {b['depth_ratio']:.2f}  (profile: {b['profile']}, flat-risk {b['flat_risk']:.2f})",
                 f"- mesh health score: {b['topology']:.2f}", "", "## Agreement with your pictures (silhouette IoU)"]
        lines += [f"- {k}: {v:.3f}" for k, v in b["per_view"].items()]
        if e["extra_views"]:
            lines += ["", "## Views projected into the texture"] + [f"- {x['label']} ({x['source']}, {x['az']:+.0f} deg)" for x in e["extra_views"]]
        if e["notes"]:
            lines += ["", "## Notes"] + [f"- {n}" for n in e["notes"]]
        lines += ["", "## Not happy with it?", "- add a picture of the weak side:  `aura-white refine <this folder> --add back=back.png`",
                  "- try other candidates:  `aura-white refine <this folder>` (new seeds) or `--pick N` for text input",
                  "- raise the quality: `--quality ultra`", f"- the outlines used are in `views/outline_*.png` - check them first"]
        (job.root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class ViewSpecSafe:
    @staticmethod
    def ok(stem: str) -> bool:
        from .views import canonical_label
        return canonical_label(stem) is not None


def proxy_for_subject(subject, kind: str, gen_opts: Optional[Dict] = None, log: Optional[Say] = None):
    """Used by `build_asset` when a picture arrives with no mesh: generate a *real* 3-D proxy for it.
    Returns (verts, faces, extra_views, meta).  Raises NoBackend (with instructions) when no true-3D generator can run."""
    from .intake import request_from_arrays
    g = dict(gen_opts or {})
    recipe = Recipe.preset(g.pop("quality", "high"))
    if g.pop("flat_ok", False):
        recipe = recipe.with_(expect_solid=False)
    opts = MakeOptions(recipe=recipe, **{k: v for k, v in g.items() if k in MakeOptions.__dataclass_fields__})
    eng = Aura3D(opts, (lambda m: (log or print)("      " + m)))
    hw = probe_hardware(opts.device)
    req = request_from_arrays(subject.rgb, subject.alpha)
    best, loop, choice = eng._shape(req, hw)
    extra = eng._extra_views(req, best.verts, best.faces, hw, None)
    meta = {"backend": choice.backend.name, "tier": choice.backend.tier, "passed": loop.passed, "reason": loop.reason,
            "depth_ratio": best.audit.depth_ratio, "view_iou": best.mean_iou, "recipe_signature": recipe.signature()}
    return best.verts, best.faces, extra, meta
