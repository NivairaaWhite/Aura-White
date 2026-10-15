"""Text -> reference picture.

A text prompt cannot go straight into a shape model, so it first becomes a picture - but not *any* picture: one a 3-D
generator can use.  The prompt is wrapped so the object is whole, centred, evenly lit and on a plain background; several
candidates are drawn and ranked by how suitable they are (is the whole object in frame? one connected silhouette? a
background that can be removed exactly?), and the best becomes the reference.  The others are kept so the person can pick
another (`aura-white refine --pick N`).

The model is `AURA_WHITE_T2I_MODEL` / `--t2i-model` (any Hugging Face text-to-image diffusers repo); the default is SDXL,
chosen because it runs on every machine the recipe supports.  Memory strategy is the planner's, not a quality choice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .hardware import ExecPlan, Hardware, plan_stage, run_with_oom_retry

Say = Callable[[str], None]
DEFAULT_T2I = "stabilityai/stable-diffusion-xl-base-1.0"
FRAMING = ("single object, full view of the whole object, centred, front view slightly turned three-quarter, "
           "soft even studio lighting, plain light grey background, no cast shadow, clean silhouette, sharp focus, highly detailed")
NEGATIVE = ("cropped, cut off, partially visible, out of frame, multiple objects, collage, text, watermark, logo, border, "
            "busy background, scenery, floor, harsh shadow, motion blur, blurry, low quality, deformed, extra limbs")


def compose_prompt(text: str, style: Optional[str] = None) -> Tuple[str, str]:
    base = text.strip().rstrip(".,")
    parts = [base] + ([style.strip()] if style else []) + [FRAMING]
    return ", ".join(parts), NEGATIVE


def score_reference(img: Image.Image) -> Tuple[float, dict]:
    """0..1: how well a picture serves as a 3-D reference (whole object, one piece, plain background, removable exactly)."""
    from .outline import classify_image, extract_outline, OutlineError
    rgb = np.asarray(img.convert("RGB"))
    _, feats = classify_image(rgb)
    try:
        o = extract_outline(img.convert("RGB"), strict=True, use_models=False)
    except OutlineError:
        return 0.0, {"reason": "no clean outline"}
    h, w = o.hard.shape
    ys, xs = np.nonzero(o.hard)
    mx, my = min(xs.min(), w - 1 - xs.max()) / w, min(ys.min(), h - 1 - ys.max()) / h
    margin = float(np.clip(min(mx, my) / 0.04, 0.0, 1.0))                 # object clear of every edge
    frac = float(o.hard.mean())
    size = float(np.clip(1.0 - abs(frac - 0.40) / 0.40, 0.0, 1.0))        # neither tiny nor filling the frame
    plain = float(np.clip(feats["border_uniform"], 0.0, 1.0))
    s = 0.35 * margin + 0.20 * size + 0.20 * plain + 0.25 * o.score
    return float(s), {"margin": margin, "size": size, "plain_background": plain, "outline": o.score, "fraction": frac}


def rank_references(images: List[Image.Image]) -> List[Tuple[int, float, dict]]:
    scored = [(i, *score_reference(im)) for i, im in enumerate(images)]
    return sorted(scored, key=lambda t: -t[1])


class TextToImage:
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("AURA_WHITE_T2I_MODEL") or DEFAULT_T2I
        self.stage = "sdxl_text2image" if "xl" in self.model.lower() else "sdxl_text2image"
        self._pipe = None

    def probe(self, hw: Hardware):
        from ..backends.base import module_missing
        miss = module_missing(["torch", "diffusers", "transformers", "accelerate"])
        if miss:
            return False, "missing Python packages: " + ", ".join(miss) + " (pip install -r requirements-3d.txt)", None
        plan = plan_stage(self.stage, hw)
        return plan.feasible, plan.note or "ready", plan

    def _load(self, plan: ExecPlan, say: Say):
        import torch
        from diffusers import AutoPipelineForText2Image
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(plan.dtype, torch.float32)
        say(f"loading {self.model} for the reference picture ({plan.offload} on {plan.device}); the first run downloads it")
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(self.model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None,
                                                             use_safetensors=True)
        except Exception:
            pipe = AutoPipelineForText2Image.from_pretrained(self.model, torch_dtype=dtype)
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

    def generate(self, text: str, n: int, size: int, seed: int, plan: ExecPlan, say: Say, style: Optional[str] = None) -> List[Image.Image]:
        import torch
        prompt, negative = compose_prompt(text, style)

        def run(p: ExecPlan):
            pipe = self._pipe or self._load(p, say)
            dev = "cuda" if p.device == "cuda" else "cpu"
            imgs = []
            for i in range(n):
                g = torch.Generator(device=dev).manual_seed(int(seed) + i)
                say(f"drawing reference picture {i + 1}/{n}")
                kw = dict(prompt=prompt, negative_prompt=negative, height=size, width=size, generator=g)
                from .shapegen import call_supported
                imgs.append(call_supported(pipe, **kw).images[0].convert("RGB"))
            return imgs

        imgs, _ = run_with_oom_retry(run, plan, say)
        return imgs

    def close(self):
        self._pipe = None
        from .hardware import free_accelerator_memory
        free_accelerator_memory()
