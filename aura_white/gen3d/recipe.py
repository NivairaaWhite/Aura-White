"""Quality recipes.

The central rule of Aura White 7: **quality is decided by the recipe, never by the hardware.**

A `Recipe` fixes every number that changes the *result* - how many denoising steps, the mesh resolution, how many
candidates are compared, how closely the shape must agree with the pictures.  The hardware planner
(`hardware.py`) is only allowed to change *how* that recipe is executed - model offloading, chunk sizes, the
numeric type - none of which changes the answer beyond float noise.  The recipe's `signature()` is written into
every job so two runs on different machines can be proven to have used the same quality settings.

Presets are named by what the person wants, not by what the machine has.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class Recipe:
    name: str = "high"
    # ---- shape generation (neural) ---------------------------------------------------------------------
    shape_steps: int = 50                 # denoising steps of the shape model
    shape_guidance: float = 5.0           # classifier-free guidance scale
    octree: int = 384                     # marching-cubes grid of the shape decoder (mesh detail)
    candidates: int = 3                   # seeds compared per round
    max_rounds: int = 3                   # refinement rounds before the best result so far is shipped
    # ---- text -> reference picture ----------------------------------------------------------------------
    t2i_candidates: int = 4               # reference pictures drawn and ranked for a text prompt
    t2i_size: int = 1024
    # ---- agreement with the supplied pictures ------------------------------------------------------------
    view_iou_target: float = 0.97         # mean silhouette IoU the finished mesh must reach in every supplied view
    fusion_grid: int = 192                # voxel grid of the silhouette-consistency fusion
    fusion_tolerance_px: float = 1.5      # silhouette slack (pixels at 256 px) before a voxel is carved away
    # ---- is it really three-dimensional --------------------------------------------------------------------
    min_depth_ratio: float = 0.22         # depth / width below this is flat risk; reaches 1.0 at half of it (a relief slab is ~0.1-0.17)
    expect_solid: bool = True             # False for coins, banners, decals: flat is then correct
    # ---- appearance of the sides and the back --------------------------------------------------------------
    paint_views: bool = True              # synthesise missing views (back / sides) from the mesh's own depth
    paint_azimuths: Tuple[float, ...] = (90.0, 180.0, 270.0)
    texture_size: int = 2048
    # ---- scoring weights --------------------------------------------------------------------------------------
    w_view: float = 1.0
    w_solid: float = 0.6
    w_topology: float = 0.3

    def signature(self) -> str:
        """Stable short hash of every quality-relevant field (the name is a label, not a setting)."""
        d = asdict(self)
        d.pop("name")
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=list).encode()).hexdigest()[:12]

    def with_(self, **kw) -> "Recipe":
        bad = [k for k in kw if k not in {f.name for f in fields(self)}]
        if bad:
            raise ValueError(f"unknown recipe field(s): {', '.join(bad)}")
        return replace(self, **kw)

    @staticmethod
    def preset(name: str = "high") -> "Recipe":
        try:
            return PRESETS[name.lower()]
        except KeyError:
            raise ValueError(f"unknown quality preset {name!r}; choose one of {', '.join(PRESETS)}") from None


PRESETS: Dict[str, Recipe] = {
    "draft": Recipe(name="draft", shape_steps=30, octree=256, candidates=1, max_rounds=1, t2i_candidates=2,
                    fusion_grid=128, texture_size=1024, paint_views=False),
    "standard": Recipe(name="standard", shape_steps=40, octree=320, candidates=2, max_rounds=2, t2i_candidates=3,
                       fusion_grid=160, texture_size=2048),
    "high": Recipe(name="high"),
    "ultra": Recipe(name="ultra", shape_steps=60, octree=512, candidates=4, max_rounds=4, t2i_candidates=6,
                    fusion_grid=256, view_iou_target=0.98, texture_size=4096),
}
