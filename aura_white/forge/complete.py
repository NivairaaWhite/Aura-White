"""Residual completion.

Every neural image-to-3D model drops something: chains, cords, thin ribbons, small gems, fur tips.  After the
mesh has been registered to the artwork (camera known), the pixels of the artwork that the mesh does NOT cover are the
missing parts.  This module builds real geometry for them:

  * wire-like residual  -> interlocking chain links / round tubes   (forge.wires)
  * blob-like residual  -> silhouette-exact relief patches (plate, or faceted "gem" roof) (forge.patch)

placed at the depth of the nearest mesh surface, so the completed model reproduces the artwork silhouette 1:1 from the
reference view.  The result is in the same frame as the mesh, ready to be appended before UV unwrapping / baking (the
artwork colours then land on the new parts automatically because they project exactly onto the residual pixels).

Pure NumPy + SciPy.  Nothing here needs a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from . import morph, patch, softraster as sr, wires as wr


@dataclass
class CompletionOptions:
    tolerance_px: float = 0.0            # 0 = auto (0.35 % of the image side)
    min_area_px: float = 0.0             # 0 = auto
    chains: bool = True
    tubes: bool = True
    patches: bool = True
    thickness: float = 0.5               # patch half-thickness as a fraction of the patch's own inscribed radius
    max_thickness_px: float = 9.0
    front_offset: float = 1.0            # parts sit this many local half-thicknesses in front of the surface
    max_faces: int = 120_000
    bg: Optional[Sequence[float]] = None


@dataclass
class CompletionResult:
    verts: np.ndarray
    faces: np.ndarray
    parts: List[dict] = field(default_factory=list)
    residual: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))


def residual_mask(ref_mask: np.ndarray, mesh_mask: np.ndarray, tol_px: float, min_area: float) -> np.ndarray:
    """Artwork pixels farther than `tol_px` from any mesh pixel (so registration slop along the silhouette is not
    mistaken for a missing part), with specks removed."""
    from scipy import ndimage as ndi

    d = morph.edt(~mesh_mask) if mesh_mask.any() else np.full(ref_mask.shape, 1e9)
    res = ref_mask & (d > tol_px)
    # grow back by the tolerance inside the artwork so the new part overlaps the mesh a little (no visible seam)
    res = ndi.binary_opening(res, np.ones((2, 2), bool))
    lab, n = ndi.label(res, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = sizes >= min_area
        keep[0] = False
        res = keep[lab]
    return res


def _flip_if_inverted(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    if len(f) == 0:
        return f
    v = v.astype(np.float64)
    vol = float(np.sum(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]]))) / 6.0)
    return f[:, [0, 2, 1]] if vol < 0 else f


def _ref_depth(mesh_mask, mesh_depth, near_idx, ys, xs, fallback):
    """Depth of the mesh surface nearest to a set of pixels."""
    if near_idx is None:
        return fallback
    iy, ix = near_idx
    d = mesh_depth[iy[ys, xs], ix[ys, xs]]
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else fallback


def complete_residual(ref_rgb: np.ndarray, ref_alpha: np.ndarray, mesh_mask: np.ndarray, mesh_depth: np.ndarray,
                      cam, opts: Optional[CompletionOptions] = None) -> CompletionResult:
    """`ref_rgb` (H,W,3) uint8, `ref_alpha` (H,W) float, `mesh_mask` (H,W) bool and `mesh_depth` (H,W) float (view-space
    depth, inf outside) are all in the reference camera `cam` (attributes w2c, f, cx, cy)."""
    from scipy import ndimage as ndi

    opts = opts or CompletionOptions()
    H, W = ref_alpha.shape
    S = float(max(H, W))
    ref = np.asarray(ref_alpha) >= 0.5
    tol = opts.tolerance_px or max(2.0, 0.0035 * S)
    min_area = opts.min_area_px or max(14.0, 5e-6 * H * W)
    res = residual_mask(ref, mesh_mask, tol, min_area)
    notes: List[str] = []
    if not res.any():
        return CompletionResult(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), [], res,
                                ["completion: the mesh already covers the artwork"])
    finite = mesh_depth[np.isfinite(mesh_depth)]
    fallback = float(np.median(finite)) if finite.size else float(np.linalg.norm(np.asarray(cam.w2c)[:3, 3]))
    near_idx = morph.nearest_index(mesh_mask) if mesh_mask.any() else None
    c2w = np.linalg.inv(np.asarray(cam.w2c, np.float64))
    view_dir = c2w[:3, :3] @ np.array([0.0, 0.0, -1.0])

    parts: List[dict] = []
    V: List[np.ndarray] = []
    F: List[np.ndarray] = []
    base = 0

    def add(v, f, info):
        nonlocal base
        if len(v) == 0 or len(f) == 0:
            return
        f = _flip_if_inverted(v, f)
        V.append(v.astype(np.float32))
        F.append(f + base)
        base += len(v)
        parts.append(info)

    # ---- wires -------------------------------------------------------------------------------------------
    rest = res.copy()
    if opts.chains or opts.tubes:
        found, used = wr.find_wires(res, ref_rgb, bg=None if opts.bg is None else np.asarray(opts.bg, np.float32))
        for w in found:
            if (w.kind == "chain" and not opts.chains) or (w.kind == "tube" and not opts.tubes):
                continue
            xs = np.clip(np.round(w.pts[:, 0]).astype(int), 0, W - 1)
            ys = np.clip(np.round(w.pts[:, 1]).astype(int), 0, H - 1)
            k = max(1, len(xs) // 6)
            w_ref = _ref_depth(mesh_mask, mesh_depth, near_idx, ys[:k], xs[:k], fallback)
            px_w = sr.pixel_size(cam, w_ref, int(H))
            w_place = w_ref - opts.front_offset * max(0.6 * w.width, 1.0) * px_w
            P = sr.unproject(cam, w.pts[:, 0] + 0.5, w.pts[:, 1] + 0.5, np.full(len(w.pts), w_place), H, W)
            width_world = w.width * px_w
            if w.kind == "chain":
                v, f, nl = wr.chain_mesh(P, width_world, view_dir)
                add(v, f, {"kind": "chain", "links": nl, "length_px": w.length, "width_px": w.width})
            else:
                v, f = wr.tube_mesh(P, 0.45 * width_world, taper=0.35)
                add(v, f, {"kind": "tube", "length_px": w.length, "width_px": w.width})
        rest = res & ~ndi.binary_dilation(used, iterations=1)

    # ---- patches -----------------------------------------------------------------------------------------
    if opts.patches and rest.any():
        rest = ndi.binary_opening(rest, np.ones((2, 2), bool))
        lab, n = ndi.label(rest, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        slices = ndi.find_objects(lab)
        budget = int(opts.max_faces)
        for i in np.argsort(sizes[1:])[::-1] + 1:
            if sizes[i] < min_area or budget <= 0:
                continue
            sl = slices[i - 1]
            pad = 4
            y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
            x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
            comp = lab[y0:y1, x0:x1] == i
            alpha = ndi.gaussian_filter(comp.astype(np.float32), 0.7)
            dt = morph.edt(comp)
            R = float(dt.max())
            area = float(comp.sum())
            bbox = float((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))
            compact = area / max(bbox, 1.0)
            half = float(np.clip(opts.thickness * R, 0.8, opts.max_thickness_px))
            if compact > 0.28 and area < 0.004 * H * W:            # gems, buttons, studs -> faceted roof
                kind = "gem"
                hmap = patch.roof_profile(dt, np.full_like(dt, max(R, 1.0)), np.full_like(dt, min(1.6 * half, 1.4 * R)))
            else:                                                    # ribbons, fins, fur -> thin plate with a bevel
                kind = "sheet"
                hmap = patch.plate_profile(dt, half, min(0.7 * R + 0.5, 4.0 + half))
            m = patch.build_relief(alpha, hmap, hmap, None, max_faces=max(2000, budget))
            if m is None:
                continue
            yy, xx = np.nonzero(comp)
            w_ref = _ref_depth(mesh_mask, mesh_depth, near_idx, yy + y0, xx + x0, fallback)
            px_w = sr.pixel_size(cam, w_ref, int(H))
            v = m.verts.astype(np.float64)
            sx = v[:, 0] + x0 + 0.5
            sy = v[:, 1] + y0 + 0.5
            w_v = w_ref - (v[:, 2] + opts.front_offset * half) * px_w
            world = sr.unproject(cam, sx, sy, w_v, H, W)
            add(world, m.faces, {"kind": kind, "area_px": area, "faces": int(len(m.faces))})
            budget -= len(m.faces)

    if not V:
        return CompletionResult(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), [], res,
                                ["completion: residual found but nothing could be built"])
    verts = np.concatenate(V, 0)
    faces = np.concatenate(F, 0)
    n_w = sum(1 for p in parts if p["kind"] in ("chain", "tube"))
    notes.append(f"completion: {n_w} wire(s), {len(parts) - n_w} patch(es), {len(faces)} faces added "
                 f"({100.0 * res.sum() / max(ref.sum(), 1):.2f}% of the artwork was missing from the mesh)")
    return CompletionResult(verts, faces, parts, res, notes)
