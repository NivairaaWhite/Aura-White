"""Aura White - lightweight single-image 3D generation.

Design rule: nothing in this package needs a compiler, a git dependency, or a pinned
third-party version. Only numpy + Pillow + torch are required; everything else is optional
and every optional import is guarded, so a Python/library update cannot break the pipeline.
"""
from __future__ import annotations

__version__ = "8.0.0"
__all__ = ["AuraWhite", "AuraConfig", "MeshResult", "__version__"]


def __getattr__(name):  # lazy: `import aura_white` must never import torch
    if name in ("AuraWhite", "MeshResult"):
        from . import pipeline

        return getattr(pipeline, name)
    if name == "AuraConfig":
        from .config import AuraConfig

        return AuraConfig
    raise AttributeError(f"module 'aura_white' has no attribute {name!r}")
