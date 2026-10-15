"""Geometry back-ends.  Each turns the framed artwork into a mesh in the canonical frame (z up, photo seen from +x).

  hunyuan3d   Hunyuan3D-2.1 shape model  - best detail (thin blades, ribbons), ~10 GB GPU memory
  instantmesh Zero123++ -> InstantMesh FlexiCubes LRM - fast, ~8 GB, also yields the six side views
  aura        the built-in TripoSR-compatible Aura White model - runs anywhere, lowest detail

All of them are loaded lazily and freed after use so a 16 GB GPU can run the whole chain."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from .base import (BackendUnavailable, GeometryResult, Progress, cuda_free_gb, find_repo, free_cuda,
                   load_state, module_missing, prepend_path)
from ._compat import import_with_stubs
from .multiview import Zero123PlusBackend


class AuraGeometry:
    name = "aura"

    def __init__(self, device: str = "auto", weights: Optional[str] = None, resolution: int = 256,
                 threshold: str = "25", engine=None):
        self.device, self.weights, self.resolution, self.threshold = device, weights, resolution, threshold
        self.engine = engine                     # an already loaded AuraWhite (tests, long-running servers)

    def status(self):
        return True, "built in"

    def generate(self, cutout: Image.Image, seed: int = 0, progress: Progress = None, resolution=None, **_):
        from ..pipeline import AuraWhite

        say = progress or (lambda *_: None)
        say("Aura White: reconstructing geometry")
        if self.weights == "random":                     # untrained network: self-tests only
            model = AuraWhite.from_random()
        else:
            model = self.engine or AuraWhite.from_pretrained(self.weights, device=self.device)
        res = model.generate(cutout, resolution=int(resolution or self.resolution), remove_bg=True,
                             threshold=self.threshold)
        return GeometryResult(res.vertices.astype(np.float32), res.faces.astype(np.int64), res.colors,
                              info={"backend": "aura", "resolution": int(resolution or self.resolution)})


class HunyuanGeometry:
    name = "hunyuan3d"
    REQUIRED = ["torch", "diffusers", "transformers", "pytorch_lightning", "einops", "omegaconf", "yaml",
                "huggingface_hub", "trimesh"]
    MODEL_ID = "tencent/Hunyuan3D-2.1"

    def __init__(self, device: str = "auto", repo=None, octree_resolution: int = 384, steps: int = 50,
                 guidance: float = 5.0, model_id: Optional[str] = None):
        self.device, self.repo = device, repo
        self.octree, self.steps, self.guidance = octree_resolution, steps, guidance
        self.model_id = model_id or self.MODEL_ID

    def status(self):
        if self._repo() is None:
            return False, "Hunyuan3D-2.1 sources not found (unzip Hunyuan3D-2.1-main into third_party/)"
        miss = module_missing(self.REQUIRED)
        if miss:
            return False, "missing Python packages: " + ", ".join(miss)
        free = cuda_free_gb()
        if free is None:
            return False, "no CUDA GPU (the shape model needs ~10 GB of GPU memory)"
        if free < 9.0:
            return False, f"only {free:.1f} GB of GPU memory free (needs ~10 GB)"
        return True, "ready"

    def _repo(self):
        return self.repo or find_repo("Hunyuan3D-2.1", "hy3dshape/hy3dshape/pipelines.py")

    def generate(self, cutout: Image.Image, seed: int = 42, progress: Progress = None, octree_resolution=None,
                 steps=None, **_):
        import torch

        ok, why = self.status()
        if not ok:
            raise BackendUnavailable(f"Hunyuan3D-2.1 is not available: {why}")
        say = progress or (lambda *_: None)
        prepend_path(self._repo() / "hy3dshape")
        try:            # optional packages Hunyuan imports at module level (pymeshlab, ...) are stubbed if missing
            Hunyuan3DDiTFlowMatchingPipeline = import_with_stubs("hy3dshape.pipelines", say).Hunyuan3DDiTFlowMatchingPipeline
        except Exception as e:
            raise BackendUnavailable(f"could not import the Hunyuan3D shape package ({type(e).__name__}: {e}). "
                                     "Install the packages listed in requirements-studio.txt.") from e
        say("loading Hunyuan3D-2.1 shape model (first run downloads ~7 GB)")
        dev = "cuda" if self.device == "auto" else self.device
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_id, device=dev, dtype=torch.float16)
        say(f"Hunyuan3D-2.1: sculpting the shape (octree {octree_resolution or self.octree}, "
            f"{steps or self.steps} steps)")
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        with torch.no_grad():
            out = pipe(image=cutout.convert("RGBA"), num_inference_steps=int(steps or self.steps),
                       guidance_scale=self.guidance, generator=gen,
                       octree_resolution=int(octree_resolution or self.octree), num_chunks=8000,
                       output_type="trimesh", enable_pbar=False)
        mesh = out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(mesh, (list, tuple)):
            mesh = mesh[0]
        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.int64)
        del pipe, out
        free_cuda()
        v = v[:, [2, 0, 1]]                     # glTF (y up, front +z)  ->  canonical (z up, front +x); proper rotation
        return GeometryResult(v, f, info={"backend": "hunyuan3d", "octree": int(octree_resolution or self.octree)})


class Hunyuan3D2Geometry:
    """Hunyuan3D-2 shape model (1.1B).  Runs in <= 8 GB and below: the memory planner picks resident / offloaded / CPU
    execution, none of which changes the result.  Adapter over `aura_white.gen3d.shapegen.Hunyuan3D2Shape`."""
    name = "hunyuan3d2"

    def __init__(self, device: str = "auto", repo=None, octree_resolution: int = 384, steps: int = 50, guidance: float = 5.0, **_):
        self.device, self.repo = device, repo
        self.octree, self.steps, self.guidance = octree_resolution, steps, guidance
        self._impl = None

    def _shape(self):
        if self._impl is None:
            from ..gen3d.shapegen import Hunyuan3D2Shape
            self._impl = Hunyuan3D2Shape(multiview=False, repo=self.repo)
        return self._impl

    def status(self):
        from ..gen3d.hardware import probe_hardware
        pr = self._shape().probe(probe_hardware(self.device), [])
        return pr.ok, pr.reason

    def generate(self, cutout: Image.Image, seed: int = 42, progress: Progress = None, octree_resolution=None, steps=None, **_):
        from ..gen3d.controller import GenCall
        from ..gen3d.hardware import probe_hardware
        from ..gen3d.intake import ViewInput
        from ..gen3d.views import ViewSpec
        say = progress or (lambda *_: None)
        hw = probe_hardware(self.device)
        impl = self._shape()
        pr = impl.probe(hw, [])
        if not pr.ok:
            raise BackendUnavailable(f"Hunyuan3D-2 is not available: {pr.reason}")
        view = ViewInput(ViewSpec.parse("front"), cutout.convert("RGBA"), np.zeros((1, 1), bool), None)
        call = GenCall(int(seed), int(octree_resolution or self.octree), int(steps or self.steps), self.guidance)
        try:
            v, f = impl.generate(call, [view], pr.plan, say)
        finally:
            impl.close()
        return GeometryResult(np.asarray(v, np.float32)[:, [2, 0, 1]], np.asarray(f, np.int64),
                              info={"backend": "hunyuan3d2", "octree": call.octree})


class InstantMeshGeometry:
    name = "instantmesh"
    REQUIRED = ["torch", "torchvision", "diffusers", "transformers", "einops", "omegaconf", "huggingface_hub"]

    def __init__(self, device: str = "auto", repo=None, config: str = "instant-mesh-large", steps: int = 75):
        self.device, self.repo, self.config, self.steps = device, repo, config, steps

    def _repo(self):
        return self.repo or find_repo("InstantMesh", "src/models/lrm_mesh.py")

    def status(self):
        if self._repo() is None:
            return False, "InstantMesh sources not found (unzip InstantMesh-main into third_party/)"
        miss = module_missing(self.REQUIRED)
        if miss:
            return False, "missing Python packages: " + ", ".join(miss)
        free = cuda_free_gb()
        if free is None:
            return False, "no CUDA GPU"
        if free < 7.0:
            return False, f"only {free:.1f} GB of GPU memory free (needs ~8 GB)"
        return True, "ready"

    def _weights(self, local: str, hub_file: str) -> str:
        repo = self._repo()
        for p in (local, str(repo / local)):
            if os.path.exists(p):
                return p
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id="TencentARC/InstantMesh", filename=hub_file, repo_type="model")

    def generate(self, cutout: Image.Image, seed: int = 42, progress: Progress = None, steps=None, **_):
        import torch
        from omegaconf import OmegaConf

        ok, why = self.status()
        if not ok:
            raise BackendUnavailable(f"InstantMesh is not available: {why}")
        say = progress or (lambda *_: None)
        repo = self._repo()
        prepend_path(repo)
        cfg = OmegaConf.load(repo / "configs" / f"{self.config}.yaml")
        dev = "cuda" if self.device == "auto" else self.device

        # 1. the white-background Zero123++ used to train the reconstructor
        unet = self._weights(cfg.infer_config.unet_path, "diffusion_pytorch_model.bin")
        z = Zero123PlusBackend(device=dev, steps=int(steps or self.steps), unet_path=unet, background="white")
        views = z.generate(cutout, seed=seed, progress=progress)
        z.unload()

        # 2. triplane reconstruction + FlexiCubes mesh extraction
        say("InstantMesh: reconstructing the mesh")
        try:
            # InstantMesh imports nvdiffrast / xatlas at module level, but mesh *extraction* needs neither
            lrm = import_with_stubs("src.models.lrm_mesh", say)
            cams = import_with_stubs("src.utils.camera_util", say)
        except Exception as e:
            raise BackendUnavailable(f"could not import InstantMesh ({type(e).__name__}: {e}). "
                                     "Install the packages listed in requirements-studio.txt.") from e
        model = lrm.InstantMesh(**cfg.model_config.params)
        ckpt = self._weights(cfg.infer_config.model_path, self.config.replace("-", "_") + ".ckpt")
        state = load_state(ckpt, "state_dict")
        state = {k[14:]: v for k, v in state.items() if k.startswith("lrm_generator.")}
        model.load_state_dict(state, strict=True)
        model = model.to(dev)
        model.init_flexicubes_geometry(torch.device(dev), fovy=30.0)
        model = model.eval()
        imgs = torch.from_numpy(np.stack(views.images)).permute(0, 3, 1, 2).unsqueeze(0).float().to(dev)
        cameras = cams.get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(dev)
        with torch.no_grad():
            planes = model.forward_planes(imgs, cameras)
            verts, faces, colors = model.extract_mesh(planes, use_texture_map=False, **cfg.infer_config)
        del model, planes
        free_cuda()
        return GeometryResult(np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64), colors,
                              views=views, absolute_scale=True, info={"backend": "instantmesh"})


def status_report():
    """{backend: (ok, reason)} for `aura-white doctor`."""
    from .external import Pixal3DGeometry, Trellis2Geometry
    from ..forge.relief import ReliefGeometry

    out = {"aura": AuraGeometry().status(), "relief": ReliefGeometry().status()}
    out["pixal3d"] = Pixal3DGeometry().status()
    out["trellis2"] = Trellis2Geometry().status()
    out["instantmesh"] = InstantMeshGeometry().status()
    out["hunyuan3d"] = HunyuanGeometry().status()
    out["zero123++"] = Zero123PlusBackend().status()
    return out


AUTO_ORDER = ("pixal3d", "trellis2", "hunyuan3d", "hunyuan3d2", "instantmesh")     # best first; the built-in model is the last resort


def make_geometry(name: str, device: str = "auto", **kw):
    """Instantiate one named back-end (no availability check)."""
    name = name.lower()
    if name in ("hunyuan", "hunyuan3d", "hunyuan3d-2.1"):
        return HunyuanGeometry(device, **{k: v for k, v in kw.items() if k in ("repo", "octree_resolution", "steps")})
    if name in ("hunyuan3d2", "hunyuan3d-2", "hunyuan2"):
        return Hunyuan3D2Geometry(device, **{k: v for k, v in kw.items() if k in ("repo", "octree_resolution", "steps")})
    if name == "instantmesh":
        return InstantMeshGeometry(device, **{k: v for k, v in kw.items() if k in ("repo", "steps")})
    if name == "pixal3d":
        from .external import Pixal3DGeometry
        return Pixal3DGeometry(device, resolution=int(kw.get("resolution") or 1024), repo=kw.get("repo"))
    if name in ("trellis2", "trellis.2", "trellis"):
        from .external import Trellis2Geometry
        return Trellis2Geometry(device, resolution=int(kw.get("resolution") or 1024), repo=kw.get("repo"))
    if name in ("relief", "exact"):
        from ..forge.relief import ReliefGeometry
        return ReliefGeometry(device)
    if name in ("aura", "builtin", "triposr"):
        return AuraGeometry(device, kw.get("weights"), engine=kw.get("engine"))
    raise ValueError(f"unknown geometry backend {name!r} (auto, pixal3d, trellis2, hunyuan3d, instantmesh, relief, aura)")


def select_geometry(name: str = "auto", device: str = "auto", **kw):
    """auto -> the best back-end that can run here (pixal3d > trellis2 > hunyuan3d > instantmesh > built-in)."""
    name = name.lower()
    if name != "auto":
        return make_geometry(name, device, **kw)
    for cand_name in AUTO_ORDER:
        try:
            cand = make_geometry(cand_name, device, **kw)
            if cand.status()[0]:
                return cand
        except Exception:
            continue
    return AuraGeometry(device, kw.get("weights"), engine=kw.get("engine"))
