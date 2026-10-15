"""Atlas baking for any builder."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..mesh.cleanup import vertex_normals
from . import atlas as A
from .bake import bake_param, reference_depth
from .layers import Chart, Layer


def bake_atlas(builder, layers0: List[Layer], max_size: int = 2048, padding: int = 8, back_mode: str = "soft",
               back_image: Optional[np.ndarray] = None, pot: bool = True):
    s, cam = builder.s, builder.cam
    charts = list(builder.charts.values())
    aw, ah = A.pack_charts(charts, max_size=max_size, padding=padding, pot=pot)
    src_alpha = builder.source_alpha() if hasattr(builder, "source_alpha") else s.alpha
    src = A.fill_outside(s.rgb, src_alpha)
    tex = np.zeros((ah, aw, 3), np.uint8)
    need_param = any(c.kind == "param" for c in charts)
    depth_ref = reference_depth(layers0, cam) if need_param else None
    for ch in charts:
        if ch.kind == "front":
            tile = A.bake_front(src, ch)
        elif ch.kind == "back":
            tile = A.bake_back(src, s.alpha, ch, ch.fill if ch.fill in ("mirror", "soft", "flat") else back_mode,
                               back_image)
        else:
            L = next(l for l in layers0 if (l.face_chart == ch.cid).any())
            vn = vertex_normals(L.verts.astype(np.float32), L.faces).astype(np.float64)
            tile = bake_param(ch, L, vn, cam, depth_ref, src, s.alpha)
        A.paste(tex, tile, ch.origin, padding, bool(ch.params.get("wrap_u")))
    return tex, (aw, ah)
