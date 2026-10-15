"""Every place where PyTorch / OS behaviour differs between versions lives here, behind feature
detection (never version-number comparisons), so upgrades don't silently break anything."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def sdpa(q, k, v):
    """Scaled dot-product attention. Uses the fused kernel when this torch has one."""
    fn = getattr(F, "scaled_dot_product_attention", None)
    if fn is not None:
        return fn(q, k, v)
    attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    return attn.softmax(dim=-1) @ v


def cross(a, b):
    fn = getattr(torch, "linalg", None)
    if fn is not None and hasattr(fn, "cross"):
        return fn.cross(a, b, dim=-1)
    return torch.cross(a, b, dim=-1)


def cache_home() -> Path:
    env = os.environ.get("AURA_WHITE_HOME")
    if env:
        return Path(env).expanduser()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "AuraWhite"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "aura_white"


def pick_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)


def pick_dtype(device: torch.device, spec: str = "auto") -> torch.dtype:
    spec = (spec or "auto").lower()
    if spec in ("fp32", "float32", "32"):
        return torch.float32
    if spec in ("fp16", "float16", "16", "half"):
        return torch.float16
    if spec in ("bf16", "bfloat16"):
        return torch.bfloat16
    # auto = fp32 everywhere: identical to the reference model and works on every backend.
    # fp16/bf16 are opt-in (--precision) for low VRAM or AMX/AVX512-BF16 CPUs.
    return torch.float32


def load_state_dict_file(path, trust_pickle: bool = False) -> dict:
    """Load a checkpoint from .safetensors, or a torch pickle (.ckpt/.pt/.pth/.bin)."""
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "This weights file is .safetensors but the 'safetensors' package is missing. "
                "Run: pip install safetensors"
            ) from e
        return dict(load_file(str(path)))
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=not trust_pickle)
    except TypeError:  # very old torch without the weights_only argument
        obj = torch.load(str(path), map_location="cpu")
    except Exception as e:
        raise RuntimeError(
            f"Could not safely read {path.name} ({type(e).__name__}: {e}). If this is the official "
            "TripoSR model.ckpt and you trust it, retry with --trust-checkpoint."
        ) from e
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"{path.name} does not contain a state dict.")
    return obj
