"""Zero123++ v1.2: one picture -> six consistent views (2x3 grid of 320 px images)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .base import BackendUnavailable, MultiViewResult, Progress, find_repo, free_cuda, module_missing

REQUIRED = ["torch", "diffusers", "transformers", "huggingface_hub"]
MODEL_ID = "sudo-ai/zero123plus-v1.2"


def _split_grid(grid: np.ndarray):
    """(960,640,3) -> six (320,320,3) views in row-major order (azimuths 30, 90, 150, 210, 270, 330)."""
    h, w = grid.shape[:2]
    s = w // 2
    return [grid[r * s:(r + 1) * s, c * s:(c + 1) * s] for r in range(h // s) for c in range(2)]


class Zero123PlusBackend:
    name = "zero123++"

    def __init__(self, device: str = "auto", steps: int = 50, unet_path: Optional[str] = None, repo=None,
                 background: str = "gray"):
        self.device_arg, self.steps, self.unet_path = device, steps, unet_path
        self.repo = repo
        self.background = background
        self.pipe = None

    # -- status --------------------------------------------------------------------------------------
    def status(self):
        miss = module_missing(REQUIRED)
        if miss:
            return False, "missing Python packages: " + ", ".join(miss)
        try:
            import torch

            if not torch.cuda.is_available():
                return False, "no CUDA GPU (Zero123++ needs ~5 GB of GPU memory)"
        except Exception as e:
            return False, str(e)
        return True, "ready"

    def _pipeline_source(self) -> str:
        repo = self.repo or find_repo("zero123plus", "diffusers-support/pipeline.py")
        if repo is not None:
            return str(repo / "diffusers-support")
        return "sudo-ai/zero123plus-pipeline"        # the same code, from the Hugging Face Hub

    def load(self, progress: Progress = None):
        if self.pipe is not None:
            return
        ok, why = self.status()
        if not ok:
            raise BackendUnavailable(f"Zero123++ is not available: {why}")
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        say = progress or (lambda *_: None)
        say("loading Zero123++ (first run downloads ~5 GB)")
        try:
            pipe = DiffusionPipeline.from_pretrained(MODEL_ID, custom_pipeline=self._pipeline_source(),
                                                     torch_dtype=torch.float16)
        except Exception as e:
            raise BackendUnavailable(
                f"could not load the Zero123++ pipeline ({type(e).__name__}: {e}).\n"
                "  This model is pinned to an older diffusers API. Known-good: diffusers==0.30.0, "
                "transformers==4.46.0 (see requirements-studio.txt).") from e
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config,
                                                                     timestep_spacing="trailing")
        if self.unet_path:
            from .base import load_state

            pipe.unet.load_state_dict(load_state(self.unet_path), strict=True)
        dev = "cuda" if self.device_arg == "auto" else self.device_arg
        self.pipe = pipe.to(dev)

    def generate(self, image_rgba: Image.Image, seed: int = 42, progress: Progress = None) -> MultiViewResult:
        import torch

        self.load(progress)
        say = progress or (lambda *_: None)
        say(f"Zero123++: imagining 6 views ({self.steps} steps)")
        img = image_rgba.convert("RGBA").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device=self.pipe.device).manual_seed(int(seed))
        with torch.no_grad():
            out = self.pipe(img, num_inference_steps=int(self.steps), generator=gen).images[0]
        grid = np.asarray(out.convert("RGB"), dtype=np.uint8)
        views = [v.astype(np.float32) / 255.0 for v in _split_grid(grid)]
        return MultiViewResult(views, grid, self.background)

    def unload(self):
        self.pipe = None
        free_cuda()
