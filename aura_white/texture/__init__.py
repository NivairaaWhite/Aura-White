"""Texturing engine: UV unwrap, decimation, camera registration, multi-view baking, normal baking.

Lazy on purpose (PEP 562): the numpy-only helpers (``unwrap``, ``decimate``, ``tangent_frames``) must be importable on a
machine without torch, exactly like ``aura_white.raster``.  Torch is imported only when a torch-backed name is used."""
from __future__ import annotations

import importlib

# torch-free: bound eagerly.  ``unwrap`` and ``decimate`` are also submodule names - binding the *functions* here (as the
# package always did) keeps ``from aura_white.texture import unwrap`` returning the function whatever imports run first.
from .decimate import decimate
from .normals import bake_normal_map, tangent_frames
from .unwrap import unwrap

# torch-backed: imported on first use
_LAZY = {
    "BakeResult": "bake", "ViewSource": "bake", "bake_texture": "bake",
    "dilate_gutters": "fill", "fill_from_neighbors": "fill",
    "Registration": "register", "align_view": "register", "estimate_foreground": "register",
    "find_orientation": "register", "make_camera": "register", "orientation_from_views": "register",
    "refine_reference_photometric": "register", "register_reference": "register",
}

__all__ = sorted(set(_LAZY) | {"decimate", "bake_normal_map", "tangent_frames", "unwrap"})


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'aura_white.texture' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value
