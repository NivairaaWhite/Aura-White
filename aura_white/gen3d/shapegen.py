"""Shape back-ends: picture(s) -> a real, closed 3-D mesh.

Every back-end here produces a *volume* (a signed-distance / occupancy field turned into a surface), never a cut-out of
the picture.  Which one runs is decided by what the machine can execute with the recipe's quality unchanged:

  universal tier (the default, runs in <= 8 GB and below, even on the processor)
      hunyuan3d2mv   Hunyuan3D-2mv, several pictures (front / left / back / right)           [neural, best]
      hunyuan3d2     Hunyuan3D-2, one picture                                                [neural, best]
  max tier (opt-in, needs a big GPU; the same recipe, a larger model)
      pixal3d / trellis2 / hunyuan3d (2.1)  via the existing back-ends
  fall-backs, used only when you allow them (they are labelled in the report, never silent)
      aura           the built-in TripoSR-compatible network (real volume, but low resolution)
      hull           a visual hull carved from >= 3 pictures (no model at all; concavities are lost)

If no true-3D back-end can run, `NoBackend` explains exactly what to install - it never hands back a slab.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .controller import GenCall
from .hardware import ExecPlan, Hardware, plan_stage, run_with_oom_retry
from .intake import ViewInput
from .views import ViewSpec

Say = Callable[[str], None]


class NoBackend(RuntimeError):
    pass


@dataclass
class Probe:
    ok: bool
    reason: str = ""
    plan: Optional[ExecPlan] = None


def call_supported(fn: Callable, *args, **kwargs):
    """Call `fn` with only the keyword arguments its signature really has (vendors rename / drop options between versions)."""
    try:
        params = inspect.signature(fn).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        pass
    return fn(*args, **kwargs)


class ShapeBackend:
    name = "base"
    tier = "universal"              # universal | max | basic | hull
    multiview = False
    stage = ""

    def probe(self, hw: Hardware, views: Sequence[ViewInput]) -> Probe:
        raise NotImplementedError

    def generate(self, call: GenCall, views: Sequence[ViewInput], plan: ExecPlan, say: Say) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def close(self):
        pass


# ---------------------------------------------------------------------------------------------------- Hunyuan3D-2
MV_LABELS = ("front", "left", "back", "right")


class Hunyuan3D2Shape(ShapeBackend):
    REQUIRED = ["torch", "diffusers", "transformers", "einops", "omegaconf", "yaml", "huggingface_hub", "trimesh", "skimage",
                "accelerate"]
    SINGLE = ("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0")
    MULTI = ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv")

    def __init__(self, multiview: bool = False, repo=None, model: Optional[Tuple[str, str]] = None):
        self.multiview = multiview
        self.name = "hunyuan3d2mv" if multiview else "hunyuan3d2"
        self.stage = "hunyuan3d2mv_shape" if multiview else "hunyuan3d2_shape"
        self.repo = repo
        self.model = model or (self.MULTI if multiview else self.SINGLE)
        self._pipe = None
        self._plan_key = None

    def _repo(self):
        from ..backends.base import find_repo
        return self.repo or find_repo("Hunyuan3D-2", "hy3dgen/shapegen/pipelines.py")

    def probe(self, hw: Hardware, views: Sequence[ViewInput]) -> Probe:
        import importlib.util
        from ..backends.base import module_missing
        if self._repo() is None and importlib.util.find_spec("hy3dgen") is None:
            return Probe(False, "Hunyuan3D-2 sources not found. Run `aura-white setup --hunyuan` (or unzip Hunyuan3D-2-main "
                                "into third_party/)")
        miss = module_missing(self.REQUIRED)
        if miss:
            return Probe(False, "missing Python packages: " + ", ".join(miss) + "  (pip install -r requirements-3d.txt)")
        if self.multiview:
            labels = {v.spec.label for v in views}
            if "front" not in labels or len(labels & set(MV_LABELS)) < 2:
                return Probe(False, "needs a front picture plus at least one of left / back / right")
        plan = plan_stage(self.stage, hw)
        if not plan.feasible:
            return Probe(False, plan.note, plan)
        return Probe(True, "ready", plan)

    def _load(self, plan: ExecPlan, say: Say):
        import torch
        from ..backends.base import prepend_path
        from ..backends._compat import import_with_stubs
        repo = self._repo()
        if repo is not None:
            prepend_path(repo)
        try:
            Pipe = import_with_stubs("hy3dgen.shapegen", say).Hunyuan3DDiTFlowMatchingPipeline
        except Exception as e:
            raise NoBackend(f"could not import hy3dgen ({type(e).__name__}: {e}). Install requirements-3d.txt") from e
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(plan.dtype, torch.float32)
        dev = "cuda" if plan.device == "cuda" else plan.device
        load_dev = "cpu" if plan.offload in ("model", "sequential") else dev
        say(f"loading {self.model[0]} ({plan.offload or 'resident'} on {plan.device}, {plan.dtype}); the first run downloads the weights")
        pipe = call_supported(Pipe.from_pretrained, self.model[0], subfolder=self.model[1], use_safetensors=True,
                              device=load_dev, dtype=dtype)
        if plan.offload in ("model", "sequential") and plan.device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:                                 # older vendor code: keep it resident rather than fail
                say(f"model offload unavailable ({type(e).__name__}) - moving the model to the GPU")
                pipe.to("cuda")
        self._pipe, self._plan_key = pipe, (plan.device, plan.offload, plan.dtype)
        return pipe

    def _ensure(self, plan: ExecPlan, say: Say):
        key = (plan.device, plan.offload, plan.dtype)
        if self._pipe is None or self._plan_key != key:
            self.close()
            self._load(plan, say)
        return self._pipe

    def generate(self, call: GenCall, views: Sequence[ViewInput], plan: ExecPlan, say: Say):
        import torch

        def run(p: ExecPlan):
            pipe = self._ensure(p, say)
            if self.multiview:
                swap = {"left": "right", "right": "left"} if call.label_swap else {}
                image = {}
                for v in views:
                    lab = swap.get(v.spec.label, v.spec.label)
                    if lab in MV_LABELS:
                        image[lab] = v.image.convert("RGBA")
            else:
                ref = next((v for v in views if v.spec.label == "front"), views[0])
                image = ref.image.convert("RGBA")
            dev = "cuda" if p.device == "cuda" else "cpu"
            gen = torch.Generator(device=dev).manual_seed(int(call.seed))
            say(f"{self.name}: sculpting (octree {call.octree}, {call.steps} steps, guidance {call.guidance:g}, seed {call.seed})")
            with torch.no_grad():
                out = call_supported(pipe, image=image, num_inference_steps=int(call.steps), guidance_scale=float(call.guidance),
                                     generator=gen, octree_resolution=int(call.octree),
                                     num_chunks=max(500, int(8000 * p.chunk_scale)), output_type="trimesh", enable_pbar=False)
            mesh = out
            while isinstance(mesh, (list, tuple)):
                mesh = mesh[0]
            return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)

        (v, f), used = run_with_oom_retry(run, plan, say)
        return v, f

    def close(self):
        self._pipe = None
        from .hardware import free_accelerator_memory
        free_accelerator_memory()


# ---------------------------------------------------------------------------------------------------- existing back-ends
class LegacyShape(ShapeBackend):
    """The back-ends the project already had (built-in network, InstantMesh, Hunyuan3D-2.1, TRELLIS.2, Pixal3D)."""

    def __init__(self, name: str, tier: str):
        self.name, self.tier = name, tier
        self.stage = "aura_triposr" if name == "aura" else ""
        self._backend = None

    def _make(self, device: str = "auto"):
        if self._backend is None:
            from ..backends.geometry import make_geometry
            self._backend = make_geometry(self.name, device)
        return self._backend

    def probe(self, hw: Hardware, views: Sequence[ViewInput]) -> Probe:
        try:
            ok, why = self._make("cpu" if hw.device == "cpu" else "auto").status()
        except Exception as e:
            return Probe(False, f"{type(e).__name__}: {e}")
        plan = plan_stage(self.stage, hw) if self.stage else ExecPlan(self.name, hw.device)
        return Probe(bool(ok), why, plan)

    def generate(self, call: GenCall, views: Sequence[ViewInput], plan: ExecPlan, say: Say):
        ref = next((v for v in views if v.spec.label == "front"), views[0])
        res = self._make().generate(ref.image.convert("RGBA"), seed=call.seed, progress=say, octree_resolution=call.octree,
                                    steps=call.steps, resolution=min(call.octree, 320))
        v = np.asarray(res.verts, np.float64)
        return v[:, [1, 2, 0]], np.asarray(res.faces, np.int64)           # legacy canonical (z up) -> y up; the controller re-checks it


# ---------------------------------------------------------------------------------------------------- visual hull
class HullShape(ShapeBackend):
    name, tier, multiview = "hull", "hull", True
    stage = ""

    def probe(self, hw: Hardware, views: Sequence[ViewInput]) -> Probe:
        az = [abs(((v.spec.az + 180.0) % 360.0) - 180.0) for v in views]
        sideways = [a for a in az if 45.0 <= a <= 135.0]
        tops = [v for v in views if abs(v.spec.el) >= 60.0]
        if len(views) < 3 or not (sideways or tops):
            return Probe(False, "a visual hull needs a front, a back and a side (or top) picture - opposite views alone carry no depth")
        return Probe(True, "ready", ExecPlan("hull", "cpu"))

    def generate(self, call: GenCall, views: Sequence[ViewInput], plan: ExecPlan, say: Say):
        from . import meshutil as M
        from .views import ViewCam
        say("hull: carving the volume from the silhouettes (no neural model)")
        n = 160
        # a unit cube; each view's mask is placed so its object fills 85 % of the frame, which is how the pictures were framed
        lo, side = np.array([-1.0, -1.0, -1.0]), 2.0
        occ = np.ones((n, n, n), bool)
        h = side / n
        ax = lo[0] + (np.arange(n) + 0.5) * h
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        for v in views:
            S = v.mask.shape[0]
            cam = ViewCam(v.spec.az, v.spec.el, S, S * 0.85 / 2.0 / 1.0, 0.0, 0.0)
            x, y, _ = cam.project(P)
            xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            ok = (xi >= 0) & (xi < S) & (yi >= 0) & (yi < S)
            inside = np.zeros(len(P), bool)
            inside[ok] = v.mask[yi[ok], xi[ok]]
            occ &= inside.reshape(n, n, n)
        if occ.sum() < 50:
            raise NoBackend("the pictures do not intersect into a volume (are they all of the same object, framed alike?)")
        return M.mesh_from_field(M.smooth_field(occ), lo, side)


# ---------------------------------------------------------------------------------------------------- Aura's own model
class AuraNativeShape(ShapeBackend):
    """Aura Native (aura_white/native): a rectified-flow generator trained by this project, CPU or GPU, no third-party weights.

    OPT-IN BY DESIGN.  A checkpoint is used only when its model card says ``approved_for_default`` (written only by
    `aura-white native approve`, which demands real training data and a passing benchmark), or when AURA_NATIVE_ALLOW_UNAPPROVED=1 is set.  The checkpoint comes from
    AURA_NATIVE_CKPT (or the `ckpt` argument).  A demo checkpoint trained on procedural shapes must never be picked silently."""
    name, tier, multiview = "aura_native", "basic", True
    stage = ""

    def __init__(self, ckpt: Optional[str] = None):
        self.ckpt = ckpt or os.environ.get("AURA_NATIVE_CKPT", "")
        self._model = None

    def _load(self):
        if self._model is None:
            from ..native.model import AuraNative
            self._model = AuraNative.load(self.ckpt, "cpu")
        return self._model

    def probe(self, hw: Hardware, views: Sequence[ViewInput]) -> Probe:
        if not self.ckpt or not os.path.isfile(self.ckpt):
            return Probe(False, "no Aura Native checkpoint (set AURA_NATIVE_CKPT to a .pt trained with `aura-white native train-*`)")
        try:
            m = self._load()
        except Exception as e:
            return Probe(False, f"checkpoint unreadable: {type(e).__name__}: {e}")
        if not m.card.get("approved_for_default") and os.environ.get("AURA_NATIVE_ALLOW_UNAPPROVED") != "1":
            return Probe(False, "this checkpoint's model card is not approved_for_default (it has not passed the benchmark gate); "
                                "set AURA_NATIVE_ALLOW_UNAPPROVED=1 to use it anyway")
        return Probe(True, "ready", ExecPlan("aura_native", "cpu"))

    def generate(self, call: GenCall, views: Sequence[ViewInput], plan: ExecPlan, say: Say):
        from ..native.model import ViewIn
        from ..native.render import prep_picture

        m = self._load()
        size = m.cfg.flow.image_size
        order = sorted(views, key=lambda v: 0 if v.spec.label == "front" else 1)
        a0 = order[0].spec.az
        vin = [ViewIn(prep_picture(np.asarray(v.image.convert("RGBA")), size), float(v.spec.az - a0), float(v.spec.el)) for v in order]
        say(f"aura native: sampling ({m.cfg.name}, {call.steps} steps)")
        g = m.generate(vin, n=1, steps=min(max(call.steps, 8), 32), cfg_scale=max(1.0, call.guidance), seed=call.seed,
                       res=min(max(call.octree, 48), 96), keep_z=False)[0]
        if len(g.faces) < 8:
            raise NoBackend("the native model produced an empty shape for these pictures")
        return np.asarray(g.verts, np.float64), np.asarray(g.faces, np.int64)


# ---------------------------------------------------------------------------------------------------- choosing
@dataclass
class Choice:
    backend: ShapeBackend
    plan: ExecPlan
    notes: List[str] = field(default_factory=list)


def candidate_backends(views: Sequence[ViewInput], tier: str = "universal", allow_basic: bool = False) -> List[ShapeBackend]:
    labels = {v.spec.label for v in views}
    mv_ok = "front" in labels and len(labels & set(MV_LABELS)) >= 2
    out: List[ShapeBackend] = []
    if tier == "max":
        out += [LegacyShape("pixal3d", "max"), LegacyShape("trellis2", "max"), LegacyShape("hunyuan3d", "max")]
    if mv_ok:
        out.append(Hunyuan3D2Shape(multiview=True))
    out.append(Hunyuan3D2Shape(multiview=False))
    if allow_basic:
        out += [HullShape(), AuraNativeShape(), LegacyShape("aura", "basic")]
    return out


def choose_backend(views: Sequence[ViewInput], hw: Hardware, tier: str = "universal", allow_basic: bool = False,
                   say: Optional[Say] = None, force: Optional[str] = None) -> Choice:
    say = say or (lambda *_: None)
    pool = candidate_backends(views, tier, allow_basic or force in ("aura", "hull"))
    if force and force != "auto":
        pool = [b for b in pool if b.name == force] or [LegacyShape(force, "max")]
    notes, reasons = [], []
    for b in pool:
        pr = b.probe(hw, views)
        if pr.ok:
            if b.tier in ("basic", "hull"):
                notes.append(f"NOTE: using the {b.tier} fall-back '{b.name}' - lower quality than the neural back-ends")
            if pr.plan and pr.plan.note:
                notes.append(f"{b.name}: {pr.plan.note}")
            return Choice(b, pr.plan, notes)
        reasons.append(f"  - {b.name}: {pr.reason}")
    hint = "" if allow_basic else "\nTo accept a lower-quality fall-back anyway add --allow-basic (built-in network / visual hull)."
    raise NoBackend("no true-3D shape generator can run here:\n" + "\n".join(reasons)
                    + "\nA flat cut-out is never produced silently." + hint)
