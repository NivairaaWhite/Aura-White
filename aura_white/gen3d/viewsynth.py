"""Synthesising the sides and back of an object from its own geometry.

Once the shape exists, the unseen sides do not have to be guessed from smoothness: the mesh knows what they look like in
depth.  For each missing azimuth the mesh is rendered as a *depth map*; a depth-conditioned image model (SDXL + depth
ControlNet) paints that exact geometry, and an IP-Adapter feeds it the reference picture, so the colours, materials and
style carry over.  The result is registered to the mesh by silhouette (`paint.register_view`) and projected into the
atlas by `paint.project_views_into_atlas`, below the weight of any real photograph.

Memory strategy is the planner's (offload / sequential); the model set is fixed, so every machine paints the same way.
The painter is an interface (`paint(...)`) - anything that returns `ExtraView`s can replace it.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .hardware import ExecPlan, Hardware, plan_stage, run_with_oom_retry
from .paint import ExtraView
from .views import ViewCam, render_maps

Say = Callable[[str], None]
VIEW_PHRASE = {"back": "seen from behind, rear view", "left": "left side view, profile", "right": "right side view, profile"}


def view_phrase(az: float) -> str:
    a = ((az + 180.0) % 360.0) - 180.0
    if abs(a) >= 135:
        return VIEW_PHRASE["back"]
    return VIEW_PHRASE["left"] if a > 0 else VIEW_PHRASE["right"]


def depth_condition(verts: np.ndarray, faces: np.ndarray, az: float, el: float, size: int, fill: float = 0.82):
    """(depth image uint8 RGB (S,S,3), mask bool) - near = bright, as depth ControlNets expect."""
    ext = float(np.ptp(verts, axis=0).max())
    c = (verts.min(0) + verts.max(0)) / 2.0
    cam0 = ViewCam(az, el, size, fill * size / max(ext, 1e-9), 0.0, 0.0)
    x0, x1, y0, y1 = cam0.extents(verts)
    cam = cam0.with_(ox=(x0 + x1) / 2.0, oy=(y0 + y1) / 2.0)
    mask, t, _, _ = render_maps(verts, faces, cam)
    d = np.zeros((size, size), np.float32)
    if mask.any():
        tv = t[mask]
        lo, hi = float(tv.min()), float(tv.max())
        d[mask] = 0.25 + 0.75 * (t[mask] - lo) / max(hi - lo, 1e-9)
    img = (np.clip(d, 0, 1) * 255).astype(np.uint8)
    return np.stack([img] * 3, -1), mask


class DepthPainter:
    BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET = "diffusers/controlnet-depth-sdxl-1.0"
    IP = ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.bin")
    stage = "sdxl_depth_paint"

    def __init__(self):
        self._pipe = None

    def probe(self, hw: Hardware):
        from ..backends.base import module_missing
        miss = module_missing(["torch", "diffusers", "transformers", "accelerate"])
        if miss:
            return False, "missing Python packages: " + ", ".join(miss), None
        plan = plan_stage(self.stage, hw)
        return plan.feasible, plan.note or "ready", plan

    def _load(self, plan: ExecPlan, say: Say):
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(plan.dtype, torch.float32)
        say("loading the depth-conditioned painter (SDXL + depth ControlNet + IP-Adapter); the first run downloads it")
        cn = ControlNetModel.from_pretrained(self.CONTROLNET, torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(self.BASE, controlnet=cn, torch_dtype=dtype, use_safetensors=True)
        pipe.load_ip_adapter(self.IP[0], subfolder=self.IP[1], weight_name=self.IP[2])
        pipe.set_ip_adapter_scale(0.85)
        if plan.device == "cuda":
            if plan.offload == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif plan.offload == "model":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        else:
            pipe.to(plan.device)
        for fn in ("enable_vae_slicing", "enable_vae_tiling"):
            try:
                getattr(pipe, fn)()
            except Exception:
                pass
        self._pipe = pipe
        return pipe

    def paint(self, verts: np.ndarray, faces: np.ndarray, reference: Image.Image, azimuths: Sequence[float], elevation: float,
              prompt: str, plan: ExecPlan, say: Say, size: int = 1024, seed: int = 0, weight: float = 0.55) -> List[ExtraView]:
        import torch
        from .shapegen import call_supported
        ref = Image.new("RGB", reference.size, (128, 128, 128))
        ref.paste(reference.convert("RGBA"), (0, 0), reference.convert("RGBA"))
        out: List[ExtraView] = []

        def run(p: ExecPlan):
            pipe = self._pipe or self._load(p, say)
            dev = "cuda" if p.device == "cuda" else "cpu"
            for i, az in enumerate(azimuths):
                depth, mask = depth_condition(verts, faces, az, elevation, size)
                say(f"painting the {view_phrase(az)} ({i + 1}/{len(azimuths)})")
                g = torch.Generator(device=dev).manual_seed(int(seed) + 17 * i)
                text = f"{prompt + ', ' if prompt else ''}{view_phrase(az)}, same object, plain grey background, soft studio lighting"
                res = call_supported(pipe, prompt=text, negative_prompt="text, watermark, blurry, deformed, different object",
                                     image=Image.fromarray(depth), ip_adapter_image=ref, controlnet_conditioning_scale=0.9,
                                     num_inference_steps=30, guidance_scale=5.0, height=size, width=size, generator=g)
                img = np.asarray(res.images[0].convert("RGB"))
                out.append(ExtraView(f"synth{az:+.0f}", float(az), float(elevation), img, mask, weight, "synthesised"))
            return out

        run_with_oom_retry(run, plan, say)
        return out

    def close(self):
        self._pipe = None
        from .hardware import free_accelerator_memory
        free_accelerator_memory()


def missing_azimuths(user_views: Sequence, wanted: Sequence[float], tol: float = 28.0) -> List[float]:
    """The wanted azimuths no real picture already covers."""
    def d(a, b):
        return abs(((a - b + 180.0) % 360.0) - 180.0)
    return [w for w in wanted if all(d(w, getattr(v, "az", getattr(getattr(v, "spec", None), "az", 1e9))) > tol for v in user_views)]
