"""Bridge to a locally-installed Blender, used as an optional retopology / bake back-end.

Blender ships QuadriFlow (quad-dominant retopology) and an OpenVDB voxel remesher, plus a mature Cycles bake
pipeline - real, maintained tools that would take a long time to reimplement and that a from-scratch mesher can
never quite match. Rather than embed Blender or link against `bpy` (which only exists inside Blender's own
interpreter), Aura White shells out to it exactly like the Pixal3D / TRELLIS.2 backends shell out to their own
Python environments: `blender --background --python runners/blender_run.py -- --task ... --in ... --out ...`.
A crash or a missing Blender install is caught and reported; nothing here is required for the rest of the pipeline.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base import Progress
from .glb import read_glb, write_glb_mesh

_RUNNER = Path(__file__).resolve().parent / "runners" / "blender_run.py"
_MIN_VERSION = (4, 0)


def _candidates():
    yield os.environ.get("AURA_WHITE_BLENDER")
    yield shutil.which("blender")
    system = platform.system()
    if system == "Windows":
        import glob

        for base in (r"C:\Program Files\Blender Foundation", r"C:\Program Files (x86)\Blender Foundation"):
            for exe in sorted(glob.glob(os.path.join(base, "Blender *", "blender.exe")), reverse=True):
                yield exe
    elif system == "Darwin":
        yield "/Applications/Blender.app/Contents/MacOS/Blender"
    else:
        for p in ("/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"):
            yield p


def find_blender() -> Optional[str]:
    """The Blender executable to use, or None (checked in order: `AURA_WHITE_BLENDER`, PATH, common install dirs)."""
    for c in _candidates():
        if c and Path(c).exists():
            return c
    return None


def _version(exe: str) -> Optional[Tuple[int, int]]:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30).stdout
        tok = out.split()[1]                        # "Blender 4.2.3\n..." -> "4.2.3"
        parts = tok.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def status() -> Tuple[bool, str]:
    exe = find_blender()
    if exe is None:
        return False, ("Blender not found (set AURA_WHITE_BLENDER to the executable, or install Blender and add it "
                       "to PATH) - AO baking and quad-remesh retopology are skipped without it, nothing else is "
                       "affected")
    v = _version(exe)
    if v is not None and v < _MIN_VERSION:
        return False, f"Blender at {exe} is {v[0]}.{v[1]}, need >= {_MIN_VERSION[0]}.{_MIN_VERSION[1]}"
    return True, exe


def _run(task: str, verts: np.ndarray, faces: np.ndarray, out_ext: str, extra: list, uv: Optional[np.ndarray] = None,
         progress: Progress = None, timeout: int = 1800):
    ok, exe_or_why = status()
    say = progress or (lambda *_: None)
    if not ok:
        say(exe_or_why)
        return None
    exe = exe_or_why
    tmp = Path(tempfile.mkdtemp(prefix="aura_blender_"))
    try:
        inp, outp = tmp / "in.glb", tmp / f"out.{out_ext}"
        write_glb_mesh(inp, verts, faces, uv=uv)
        cmd = [exe, "--background", "--factory-startup", "--python", str(_RUNNER), "--",
              "--task", task, "--in", str(inp), "--out", str(outp)] + [str(x) for x in extra]
        say(f"blender: running {task}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not outp.exists():
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
            say(f"blender {task} failed (exit {proc.returncode}): {tail}")
            return None
        return outp, tmp
    except Exception as e:
        say(f"blender {task} failed: {type(e).__name__}: {e}")
        return None


def bake_ao_curvature(verts: np.ndarray, faces: np.ndarray, uv_v: np.ndarray, uv_f: np.ndarray, uv: np.ndarray,
                      size: int = 1024, samples: int = 64, progress: Progress = None) -> Optional[np.ndarray]:
    """Bake ambient occlusion into Aura White's own UV layout with Cycles. Returns a (size,size) uint8 grayscale
    map, or None (with a message sent to `progress`) if Blender isn't available or the bake fails."""
    from PIL import Image

    res = _run("ao_bake", uv_v, uv_f, "png", ["--size", size, "--samples", samples], uv=uv, progress=progress)
    if res is None:
        return None
    outp, tmp = res
    try:
        img = np.asarray(Image.open(outp).convert("L"))
        if img.shape != (size, size):
            img = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))
        return img
    except Exception as e:
        (progress or (lambda *_: None))(f"blender ao_bake: could not read the result: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def quad_remesh(verts: np.ndarray, faces: np.ndarray, target_faces: int = 20000, preserve_boundary: bool = True,
               progress: Progress = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """QuadriFlow quad-dominant retopology (triangulated on export, so callers still get triangles), preserving
    the silhouette. Returns (verts, faces) in the input's own frame, or None if Blender isn't available."""
    extra = ["--target-faces", target_faces] + (["--preserve-boundary"] if preserve_boundary else [])
    res = _run("quad_remesh", verts, faces, "glb", extra, progress=progress)
    if res is None:
        return None
    outp, tmp = res
    try:
        v, f = read_glb(outp)
        return v, f
    except Exception as e:
        (progress or (lambda *_: None))(f"blender quad_remesh: could not read the result: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def voxel_remesh(verts: np.ndarray, faces: np.ndarray, voxel_size: float,
                 progress: Progress = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """OpenVDB voxel remesh - good for closing complicated holes / merging disjoint parts into one shell before a
    quad_remesh pass. `voxel_size` is in the mesh's own units (Aura White meshes are radius ~0.5-1)."""
    res = _run("voxel_remesh", verts, faces, "glb", ["--voxel-size", voxel_size], progress=progress)
    if res is None:
        return None
    outp, tmp = res
    try:
        v, f = read_glb(outp)
        return v, f
    except Exception as e:
        (progress or (lambda *_: None))(f"blender voxel_remesh: could not read the result: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
