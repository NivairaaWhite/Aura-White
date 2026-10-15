"""Layers and charts: the shared vocabulary of the asset builder.

A *layer* is one piece of geometry in world space (y up, source-image pixels as units) with a UV *per face corner* in
chart-local pixels.  A *chart* is a rectangle of texture with a rule for how its texels are produced (``kind``):

    front    texels are the source picture itself (a crop of it): the reference view is reproduced pixel-exactly
    back     texels are the far side of a relief: mirrored picture / softened mirror / a back view you supplied
    param    analytic surface (sphere, revolve): each texel is baked by projecting its 3-D point into the picture,
             with a depth test; texels the picture cannot see are filled by the chart's ``fill`` rule

Corner UVs (instead of per-vertex UVs) make seams automatic: ``assemble`` splits vertices wherever the UVs differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Chart:
    cid: int
    width: int                       # chart-local size in texels at scale 1 (source-pixel units)
    height: int
    kind: str                        # front | back | param
    crop: Tuple[int, int, int, int] = (0, 0, 0, 0)    # front/back: source rectangle (x0, y0, x1, y1)
    fill: str = "through"            # param: through | rot180 | planar180 | fold2 | nearest ;   back: mirror | soft | flat
    params: Dict[str, object] = field(default_factory=dict)
    # filled in by the atlas packer:
    density: float = 1.0             # relative texel density wish (the back of a relief needs far fewer texels)
    origin: Tuple[int, int] = (0, 0)
    scale: float = 1.0


@dataclass
class Layer:
    name: str
    verts: np.ndarray                # (V,3) float64 world
    faces: np.ndarray                # (F,3) int64, outward (counter-clockwise from outside)
    corner_uv: np.ndarray            # (F,3,2) float32, chart-local px
    face_chart: np.ndarray           # (F,) int32 -> Chart.cid
    role: str = "body"               # body | skin | card | floor ...
    info: Dict[str, object] = field(default_factory=dict)
    face_side: Optional[np.ndarray] = None   # (F,) int8: which half of a folded chart (fill 'fold2') a face belongs to

    def sides(self) -> np.ndarray:
        return self.face_side if self.face_side is not None else np.zeros(len(self.faces), np.int8)


def merge_layers(layers: Sequence[Layer]) -> Layer:
    v, f, uv, fc, sd, off = [], [], [], [], [], 0
    for L in layers:
        v.append(L.verts)
        f.append(L.faces + off)
        uv.append(L.corner_uv)
        fc.append(L.face_chart)
        sd.append(L.sides())
        off += len(L.verts)
    return Layer("merged", np.concatenate(v), np.concatenate(f), np.concatenate(uv), np.concatenate(fc),
                 face_side=np.concatenate(sd))


def assemble(layer: Layer, charts: Dict[int, Chart], atlas_size: Tuple[int, int],
             normals: Optional[np.ndarray] = None):
    """-> (verts (N,3) float32, faces (F,3), uv (N,2) float32 in [0,1] glTF convention, normals (N,3) float32,
    src_index (N,) original vertex of each output vertex).  Vertices are split wherever corner UV or chart differ."""
    from ..mesh.cleanup import vertex_normals
    aw, ah = atlas_size
    F = len(layer.faces)
    uv_atlas = np.empty((F, 3, 2), np.float64)
    for cid, ch in charts.items():
        sel = layer.face_chart == cid
        if sel.any():
            uv_atlas[sel] = (layer.corner_uv[sel].astype(np.float64) * ch.scale + np.array(ch.origin, np.float64))
    uv_atlas[..., 0] /= aw
    uv_atlas[..., 1] /= ah
    if normals is None:
        normals = vertex_normals(layer.verts.astype(np.float32), layer.faces)
    key = np.concatenate([layer.faces.reshape(-1, 1).astype(np.float64),
                          np.round(uv_atlas.reshape(-1, 2), 6)], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    src = uniq[:, 0].astype(np.int64)
    return (layer.verts[src].astype(np.float32), inv.reshape(F, 3).astype(np.int64),
            uniq[:, 1:].astype(np.float32), np.asarray(normals, np.float32)[src], src)
