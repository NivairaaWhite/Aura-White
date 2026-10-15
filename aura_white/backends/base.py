"""Shared types and helpers for the heavy model back-ends (Zero123++, InstantMesh, Hunyuan3D-2.1)."""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

Progress = Optional[Callable[[str], None]]


class BackendUnavailable(RuntimeError):
    """Raised with an actionable message when a back-end cannot run (missing repo, package, GPU, ...)."""


@dataclass
class MultiViewResult:
    images: List[np.ndarray]                 # 6 x (h,w,3) float32 sRGB [0,1], Zero123++ grid order
    grid: Optional[np.ndarray] = None        # the raw (960,640,3) uint8 generation, for the QA sheet
    background: str = "gray"                 # flat background colour family of the generated views


@dataclass
class GeometryResult:
    verts: np.ndarray                        # (V,3) float32, canonical frame (z up, photo seen from +x)
    faces: np.ndarray                        # (F,3) int64, counter-clockwise from outside
    colors: Optional[np.ndarray] = None      # optional (V,3) uint8 vertex colours
    views: Optional[MultiViewResult] = None  # side views produced together with the geometry
    absolute_scale: bool = False             # True: mesh is expressed in Zero123++ world units (radius-4 cameras)
    info: Dict[str, object] = field(default_factory=dict)


def module_missing(modules: Sequence[str]) -> List[str]:
    out = []
    for m in modules:
        try:
            if importlib.util.find_spec(m.split(".")[0]) is None:
                out.append(m)
        except Exception:
            out.append(m)
    return out


def third_party_roots() -> List[Path]:
    roots = []
    env = os.environ.get("AURA_WHITE_THIRD_PARTY")
    if env:
        roots += [Path(p) for p in env.split(os.pathsep) if p]
    here = Path(__file__).resolve().parents[2]
    roots += [here / "third_party", Path.cwd() / "third_party", here.parent, Path.cwd()]
    seen, out = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def find_repo(prefix: str, marker: str) -> Optional[Path]:
    """Locate an unpacked upstream repository, e.g. find_repo('InstantMesh', 'src/models/lrm_mesh.py').
    Folder names like InstantMesh-main / InstantMesh are both accepted."""
    for root in third_party_roots():
        if not root.is_dir():
            continue
        cands = [root / prefix] + sorted(root.glob(prefix + "-*")) + sorted(root.glob(prefix + "_*"))
        for c in cands:
            if (c / marker).exists():
                return c
            # zip files unpack to <name>-main/<name>-main/
            for sub in (c.glob("*") if c.is_dir() else []):
                if (sub / marker).exists():
                    return sub
    return None


def free_cuda():
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def cuda_free_gb(device=None) -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(device)
            return free / 1024 ** 3
    except Exception:
        pass
    return None


def prepend_path(p: Path):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_state(path, key: Optional[str] = None):
    """torch.load that prefers the safe path and only falls back to full unpickling for checkpoints published
    by the model authors themselves (Lightning .ckpt files contain non-tensor objects)."""
    import torch

    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        d = torch.load(path, map_location="cpu", weights_only=False)
    return d[key] if key else d
