"""Aura White 7 - the true-3D engine (text / one picture / several pictures -> a real, closed, textured asset).

Importing this package never imports torch; heavy stacks are loaded only by the stages that need them.
"""
from __future__ import annotations

from .recipe import PRESETS, Recipe

__all__ = ["Recipe", "PRESETS", "Aura3D", "MakeOptions", "Result", "NoBackend", "OutlineError", "extract_outline"]


def __getattr__(name):
    if name in ("Aura3D", "MakeOptions", "Result"):
        from . import engine
        return getattr(engine, name)
    if name == "NoBackend":
        from .shapegen import NoBackend
        return NoBackend
    if name in ("OutlineError", "extract_outline"):
        from . import outline
        return getattr(outline, name)
    raise AttributeError(f"module 'aura_white.gen3d' has no attribute {name!r}")
