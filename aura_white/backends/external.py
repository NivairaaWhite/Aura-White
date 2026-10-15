"""Geometry back-ends that run in their OWN Python environment (subprocess), so their CUDA extensions, pinned torch
versions and compiled wheels can never conflict with Aura White or with each other.

  pixal3d   TencentARC/Pixal3D  - pixel-aligned image-to-3D on the TRELLIS.2 backbone; the geometry lines up with
            the input picture, which is what makes a 1:1 result reachable.  `python inference.py --image --output`
  trellis2  microsoft/TRELLIS.2 (4B, O-Voxel, 512^3-1536^3, MIT) via our runner script (README API)

Configuration (environment variables, or constructor arguments):
  AURA_WHITE_PIXAL3D_DIR / AURA_WHITE_TRELLIS2_DIR       repository folder   (default: third_party/<name>)
  AURA_WHITE_PIXAL3D_PYTHON / AURA_WHITE_TRELLIS2_PYTHON python of that env  (default: this interpreter)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .base import BackendUnavailable, GeometryResult, Progress, cuda_free_gb, find_repo
from .glb import read_glb


class SubprocessGeometry:
    name = "external"
    repo_prefix = ""
    repo_marker = ""
    env_prefix = ""
    min_vram_gb = 16.0
    timeout_s = 3600

    def __init__(self, device: str = "auto", repo: Optional[str] = None, python: Optional[str] = None,
                 resolution: int = 1024, low_vram: Optional[bool] = None, **_):
        self.device, self.resolution, self.low_vram = device, int(resolution), low_vram
        self._repo_arg = repo
        self._python_arg = python

    # -- discovery -----------------------------------------------------------------------------------
    def _repo(self) -> Optional[Path]:
        p = self._repo_arg or os.environ.get(f"AURA_WHITE_{self.env_prefix}_DIR")
        if p:
            return Path(p) if (Path(p) / self.repo_marker).exists() else None
        return find_repo(self.repo_prefix, self.repo_marker)

    def _python(self) -> str:
        return self._python_arg or os.environ.get(f"AURA_WHITE_{self.env_prefix}_PYTHON") or sys.executable

    def status(self):
        repo = self._repo()
        if repo is None:
            return False, (f"{self.name}: repository not found (run `aura-white setup --repos {self.name}` or set "
                           f"AURA_WHITE_{self.env_prefix}_DIR)")
        py = self._python()
        if not (shutil.which(py) or Path(py).exists()):
            return False, f"{self.name}: python {py!r} not found (set AURA_WHITE_{self.env_prefix}_PYTHON)"
        free = cuda_free_gb()
        if free is None and not (self._python_arg or os.environ.get(f"AURA_WHITE_{self.env_prefix}_PYTHON")):
            return False, f"{self.name}: no CUDA GPU is visible (or set AURA_WHITE_{self.env_prefix}_PYTHON to the env that has one)"
        if free is not None and free < self.min_vram_gb * 0.6:
            return False, f"{self.name}: only {free:.0f} GB of GPU memory is free (needs about {self.min_vram_gb:.0f} GB)"
        return True, str(repo)

    # -- to implement --------------------------------------------------------------------------------
    def command(self, repo: Path, image: Path, out: Path, seed: int, **kw) -> List[str]:
        raise NotImplementedError

    # -- run -----------------------------------------------------------------------------------------
    def generate(self, cutout: Image.Image, seed: int = 42, progress: Progress = None, **kw) -> GeometryResult:
        ok, why = self.status()
        if not ok:
            raise BackendUnavailable(why)
        say = progress or (lambda *_: None)
        repo = self._repo()
        tmp = Path(tempfile.mkdtemp(prefix=f"aura_{self.name}_"))
        try:
            img_path, out_path = tmp / "input.png", tmp / "output.glb"
            cutout.convert("RGBA").save(img_path)
            cmd = self.command(repo, img_path, out_path, int(seed), **kw)
            say(f"{self.name}: running in its own environment ({Path(self._python()).name})")
            env = dict(os.environ)
            env.setdefault("ATTN_BACKEND", "sdpa")
            env["PYTHONUNBUFFERED"] = "1"
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True, timeout=self.timeout_s)
            if proc.returncode != 0 or not out_path.exists():
                tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-12:])
                raise BackendUnavailable(f"{self.name} failed (exit {proc.returncode}):\n{tail}")
            v, f = read_glb(out_path)
            say(f"{self.name}: {len(f):,} faces in {time.time() - t0:.0f}s")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        v = np.asarray(v, np.float32)[:, [2, 0, 1]]           # glTF (y up, front +z) -> canonical (z up, front +x)
        return GeometryResult(v, np.asarray(f, np.int64), info={"backend": self.name, "resolution": self.resolution})


class Pixal3DGeometry(SubprocessGeometry):
    name = "pixal3d"
    repo_prefix = "Pixal3D"
    repo_marker = "inference.py"
    env_prefix = "PIXAL3D"
    min_vram_gb = 24.0

    def command(self, repo, image, out, seed, **kw):
        cmd = [self._python(), "inference.py", "--image", str(image), "--output", str(out)]
        low = self.low_vram
        if low is None:
            free = cuda_free_gb()
            low = free is not None and free < 32.0
        if low:
            cmd.append("--low_vram")
        cmd += ["--resolution", str(self.resolution)]
        return cmd


class Trellis2Geometry(SubprocessGeometry):
    name = "trellis2"
    repo_prefix = "TRELLIS.2"
    repo_marker = "trellis2/pipelines/__init__.py"
    env_prefix = "TRELLIS2"
    min_vram_gb = 24.0

    def command(self, repo, image, out, seed, **kw):
        runner = Path(__file__).resolve().parent / "runners" / "trellis2_run.py"
        ptype = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}.get(self.resolution, "1024_cascade")
        return [self._python(), str(runner), "--repo", str(repo), "--image", str(image), "--output", str(out),
                "--seed", str(seed), "--pipeline-type", ptype]
