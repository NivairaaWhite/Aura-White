"""Game-ready asset packs from one picture: ``aura-white asset picture.png`` (see README, "Game assets (v6)").

Everything here needs only numpy, Pillow and scipy; xatlas, fast-simplification and trimesh are optional and guarded.
The public entry points are loaded lazily so ``import aura_white.asset`` stays cheap.
"""
from __future__ import annotations

__all__ = ["AssetOptions", "build_asset"]


def __getattr__(name):
    if name in __all__:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
