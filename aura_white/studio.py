"""Aura White Studio - artwork -> game-ready textured mesh.

    artwork --> [geometry: Hunyuan3D-2.1 | InstantMesh | built-in Aura White]
            --> [side views: Zero123++]                       (optional; the far side of the object)
            --> clean + decimate + UV unwrap (xatlas | built-in)
            --> register the original artwork to the mesh     (silhouette matching)
            --> multi-view texture bake at up to 4096 px      (the artwork's own pixels wherever it can see)
            --> normal map from the high-poly surface
            --> GLB / OBJ+MTL with albedo (+ normal) textures

Every heavy stage is optional and degrades gracefully: without a GPU the built-in geometry model and the
NumPy/PyTorch texturing still produce a UV-textured mesh.  Nothing here needs a compiler."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .backends import AuraGeometry, BackendUnavailable, Zero123PlusBackend, select_geometry
from .backends.base import GeometryResult, free_cuda
from .image.preprocess import prepare_image
from .mesh.cleanup import keep_large_components, signed_volume, vertex_normals
from .mesh.export import glb_bytes
from .mesh.textured import glb_textured_bytes, to_up_axis, write_obj_textured
from .raster import (ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS, Camera, get_raster, normalize_mesh, orbit_camera,
                     transform_camera)

Progress = Optional[Callable[[str, float, str], None]]

PRESETS: Dict[str, Dict[str, object]] = {
    "draft":    dict(texture_size=1024, target_faces=25_000, octree=256, geo_steps=30, mv_steps=30, normal_map=False,
                     blender_bake=False, lods=0),
    "standard": dict(texture_size=2048, target_faces=60_000, octree=384, geo_steps=50, mv_steps=50, normal_map=True,
                     blender_bake=False, lods=0),
    "high":     dict(texture_size=4096, target_faces=100_000, octree=384, geo_steps=50, mv_steps=75, normal_map=True,
                     blender_bake=True, lods=3),
    "ultra":    dict(texture_size=4096, target_faces=200_000, octree=512, geo_steps=60, mv_steps=75, normal_map=True,
                     blender_bake=True, lods=4),
}
STAGES = ["artwork", "geometry", "side views", "registration", "UV atlas", "texture", "normal map", "preview"]


@dataclass
class StudioOptions:
    geometry: str = "auto"              # auto | hunyuan3d | instantmesh | aura
    multiview: str = "auto"             # auto | on | off
    quality: str = "high"               # draft | standard | high | ultra
    texture_size: Optional[int] = None
    target_faces: Optional[int] = None
    octree: Optional[int] = None
    geo_steps: Optional[int] = None
    mv_steps: Optional[int] = None
    normal_map: Optional[bool] = None
    blender_bake: Optional[bool] = None  # bake real ambient occlusion with a local Blender install (needs Blender)
    lods: Optional[int] = None           # extra decimated LOD levels to generate alongside the main mesh (0 = off)
    collision: bool = False              # also generate a convex-hull collision mesh
    unwrap: str = "auto"                # auto | xatlas | builtin
    seed: int = 42
    reference_weight: float = 8.0       # how strongly the artwork's own pixels win over generated views
    generated_weight: float = 1.0
    exponent: float = 4.0               # view-angle sharpness of the blend
    reference_max_side: int = 4096      # keep the artwork at up to this size for the texture
    roughness: float = 0.7
    metallic: float = 0.0
    min_component: float = 0.002        # drop floating fragments smaller than this share of the largest part
    matting: str = "auto"               # auto | exact | rembg | border
    best_of: Tuple[str, ...] = ()       # e.g. ("pixal3d", "trellis2", "hunyuan3d"): run all, keep the best silhouette match
    repair: bool = True                 # topology repair (non-manifold edges/vertices, holes, dust)
    snap: bool = True                   # pull the mesh outline onto the artwork outline (kept only if it helps)
    complete: bool = True               # build the parts of the artwork the mesh is missing (chains, ribbons, gems ...)
    complete_tubes: bool = True
    working_res: int = 1024             # resolution of the snap / completion analysis
    back: str = "auto"                  # auto | mirror | off: texture the hidden back with the mirrored artwork
                                        # (auto = only for the depth-symmetric relief back-end)
    photometric_refine: bool = True     # refine the artwork camera so artwork and side views agree on colours
    fidelity: str = "source_locked"     # source_locked | standard: make the supplied view the visual authority
    fidelity_resolution: int = 2048     # longest side of the source-locked silhouette (never upscaled)
    fidelity_point_step: int = 8        # interior sampling spacing in pixels
    fidelity_boundary_step: int = 1     # boundary sampling spacing (1 = every pixel edge)
    fidelity_target_iou: float = 0.999  # source-locked: silhouette match this pipeline tries to reach
    clean_rim: bool = False             # source-locked: replace the pale old-background blend on the outline by object colour
                                        # (nicer on dark viewers, but alters real white/silver edge detail; default = pixel-exact)
    strict_fidelity: bool = False       # raise instead of shipping a below-target asset (source_locked only)
    refine_iterations: int = 2          # extra render->compare->re-snap passes to close the gap to the artwork
    refine_target_iou: float = 0.985    # stop refining once the rendered silhouette reaches this vs. the artwork

    def resolved(self, on_gpu: bool) -> Dict[str, object]:
        if self.quality not in PRESETS:
            raise ValueError(f"quality must be one of {list(PRESETS)}")
        p = dict(PRESETS[self.quality])
        if not on_gpu:                                   # keep CPU runs practical
            p["texture_size"] = min(int(p["texture_size"]), 2048)
            p["target_faces"] = min(int(p["target_faces"]), 60_000)
        for k in ("texture_size", "target_faces", "octree", "geo_steps", "mv_steps", "normal_map", "blender_bake",
                  "lods"):
            v = getattr(self, k)
            if v is not None:
                p[k] = v
        return p


@dataclass
class TexturedResult:
    verts: np.ndarray                    # (V,3) canonical frame (z up, artwork seen from +x), radius-1 normalised
    faces: np.ndarray                    # (F,3) int64
    uv: np.ndarray                       # (V,2) glTF convention (v downwards)
    normals: np.ndarray                  # (V,3)
    tangents: Optional[np.ndarray]       # (V,4) or None
    albedo: np.ndarray                   # (T,T,3) uint8
    normal_map: Optional[np.ndarray]     # (T,T,3) uint8 or None
    orm: Optional[np.ndarray] = None     # (T,T,3) uint8: occlusion / roughness / metallic (glTF channel order)
    unlit: bool = False                  # export KHR_materials_unlit so viewers cannot relight the source pixels
    frame: Tuple[np.ndarray, float] = (np.zeros(3), 1.0)   # (centre, scale): normalised = (original - centre) * scale
    report: Dict[str, object] = field(default_factory=dict)
    previews: Dict[str, np.ndarray] = field(default_factory=dict)
    roughness: float = 0.7
    metallic: float = 0.0
    lods: List[object] = field(default_factory=list)          # List[mesh.lod.LODLevel], LOD0 first (full detail)
    collision: Optional[object] = None                        # mesh.lod.CollisionMesh, if requested

    def glb(self, up: str = "y", scale: float = 1.0) -> bytes:
        tg = None
        if self.tangents is not None:
            tg = np.concatenate([to_up_axis(self.tangents[:, :3], up), self.tangents[:, 3:4]], axis=1)
        return glb_textured_bytes(to_up_axis(self.verts, up) * scale, self.faces, self.uv, self.albedo,
                                  to_up_axis(self.normals, up), self.normal_map, tg, self.metallic, self.roughness,
                                  orm=self.orm, unlit=self.unlit)

    def save_all(self, directory, stem: str = "model", formats: Sequence[str] = ("glb", "obj"), up: str = "y",
                 scale: float = 1.0) -> Dict[str, Path]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        out: Dict[str, Path] = {}
        for fmt in formats:
            if fmt == "glb":
                (d / f"{stem}.glb").write_bytes(self.glb(up, scale))
                out["glb"] = d / f"{stem}.glb"
            elif fmt == "obj":
                out["obj"] = write_obj_textured(d / f"{stem}.obj", to_up_axis(self.verts, up) * scale, self.faces,
                                                self.uv, self.albedo, to_up_axis(self.normals, up), self.normal_map)
            else:
                raise ValueError(f"textured export supports glb and obj, not {fmt!r}")
        Image.fromarray(self.albedo).save(d / f"{stem}_albedo.png")
        out["albedo"] = d / f"{stem}_albedo.png"
        if self.normal_map is not None:
            Image.fromarray(self.normal_map).save(d / f"{stem}_normal.png")
            out["normal"] = d / f"{stem}_normal.png"
        if self.orm is not None:
            Image.fromarray(self.orm).save(d / f"{stem}_orm.png")
            out["orm"] = d / f"{stem}_orm.png"
        for i, lvl in enumerate(self.lods):
            gv, gf = to_up_axis(np.asarray(lvl.verts, np.float32), up) * scale, np.asarray(lvl.faces, np.int64)
            p = d / f"{stem}_{lvl.name.lower()}.glb"
            p.write_bytes(glb_bytes(gv, gf))
            out[lvl.name.lower()] = p
        if self.collision is not None:
            gv = to_up_axis(np.asarray(self.collision.verts, np.float32), up) * scale
            p = d / f"{stem}_collision.glb"
            p.write_bytes(glb_bytes(gv, np.asarray(self.collision.faces, np.int64)))
            out["collision"] = p
        for name, img in self.previews.items():
            arr = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8) if img.dtype != np.uint8 else img
            Image.fromarray(arr).save(d / f"{stem}_{name}.png")
            out[name] = d / f"{stem}_{name}.png"
        (d / f"{stem}_report.json").write_text(json.dumps(self.report, indent=2, default=str), encoding="utf-8")
        out["report"] = d / f"{stem}_report.json"
        return out


class AuraStudio:
    def __init__(self, geometry=None, multiview=None, options: Optional[StudioOptions] = None, device: str = "auto",
                 raster: str = "auto", weights: Optional[str] = None, engine=None):
        """`geometry` / `multiview` may be back-end objects (anything with .generate()), names, or None (auto)."""
        self.options = options or StudioOptions()
        self.device = device
        self._geometry, self._multiview, self.weights, self.engine = geometry, multiview, weights, engine
        self.raster = get_raster(device, raster)

    # -- helpers -------------------------------------------------------------------------------------
    def _geometry_backend(self, o: StudioOptions):
        g = self._geometry if self._geometry is not None else o.geometry
        if isinstance(g, str):
            return select_geometry(g, self.device, weights=self.weights, engine=self.engine)
        return g

    def _multiview_backend(self, o: StudioOptions, mv_steps: int):
        m = self._multiview if self._multiview is not None else o.multiview
        if m in ("off", False):
            return None
        if isinstance(m, str):
            z = Zero123PlusBackend(self.device, steps=mv_steps)
            ok, why = z.status()
            if ok:
                return z
            if m == "on":
                raise BackendUnavailable(f"Zero123++ side views were required but are unavailable: {why}")
            return None
        return m

    def _refine_to_artwork(self, nv, faces, reg, ref_rgb, ref_alpha, o, notes):
        """Silhouette snap + residual completion in the artwork's camera (NumPy/SciPy, no GPU needed)."""
        from PIL import Image as _Image
        from scipy.spatial import cKDTree

        from .forge import complete as fc, snap as fs, softraster as sr
        from .texture import decimate

        work = int(min(o.working_res, ref_alpha.shape[0]))
        ref_a = np.asarray(_Image.fromarray((ref_alpha * 255 + 0.5).astype(np.uint8)).resize((work, work), _Image.BILINEAR),
                           np.float32) / 255.0
        ref_c = np.asarray(_Image.fromarray((ref_rgb * 255 + 0.5).astype(np.uint8)).resize((work, work), _Image.LANCZOS))
        pv, pf, _ = decimate(nv, faces, 60_000)
        pv = np.asarray(pv, np.float32)
        info: Dict[str, object] = {}
        if o.snap:
            res = fs.snap_to_silhouette(pv, np.asarray(pf, np.int64), reg.camera, ref_a)
            info["snap"] = dict(iou_before=round(res.iou_before, 4), iou_after=round(res.iou_after, 4), moved=res.moved)
            if res.iou_after > res.iou_before + 0.002:
                disp = (res.verts - pv).astype(np.float32)
                d, idx = cKDTree(pv).query(nv, k=3)
                wgt = 1.0 / (d + 1e-6 * max(float(d.max()), 1e-9) + 1e-12)
                wgt /= wgt.sum(1, keepdims=True)
                nv = (nv + (disp[idx] * wgt[..., None]).sum(1)).astype(np.float32)
                pv = res.verts.astype(np.float32)
                notes.append(f"silhouette snap: outline IoU {res.iou_before:.3f} -> {res.iou_after:.3f}")
            else:
                info["snap"]["applied"] = False
        comp = None
        if o.complete:
            mask, depth, _t = sr.render_camera(reg.camera, pv, np.asarray(pf, np.int64), work)
            comp = fc.complete_residual(ref_c, ref_a, mask, depth, reg.camera,
                                        fc.CompletionOptions(tubes=o.complete_tubes))
            notes += comp.notes
            from collections import Counter
            info["completion"] = dict(faces=comp.n_faces, parts=dict(Counter(p_["kind"] for p_ in comp.parts)))
        return nv, comp, info

    # -- main entry ----------------------------------------------------------------------------------
    def generate(self, image, progress: Progress = None, **overrides) -> TexturedResult:
        from .texture import (align_view, bake_normal_map, bake_texture, decimate, estimate_foreground,
                              find_orientation, orientation_from_views, refine_reference_photometric,
                              register_reference, tangent_frames, unwrap, ViewSource)
        from .texture.preview import render_textured, turntable_sheet

        o = replace(self.options, **overrides) if overrides else self.options
        on_gpu = self.raster.device.type == "cuda"
        cfg = o.resolved(on_gpu)
        if o.fidelity not in ("standard", "source_locked"):
            raise ValueError("fidelity must be 'standard' or 'source_locked'")
        no_scipy = False
        if o.fidelity == "source_locked":
            try:
                import scipy  # noqa: F401  (triangulation + distance transforms of the exact-look surface)
            except ImportError:
                o, no_scipy = replace(o, fidelity="standard"), True
        if o.fidelity == "source_locked":
            # the reference view is the visual authority: nothing generated is allowed to override it
            o = replace(o, snap=True, complete=False, photometric_refine=False,
                        reference_weight=100.0, generated_weight=0.0, unwrap="builtin")
        timings: Dict[str, float] = {}
        notes: List[str] = ["exact-look (source_locked) mode needs scipy - fell back to standard: pip install scipy"] \
            if no_scipy else []
        t_all = time.time()

        def say(stage: int, msg: str):
            if progress:
                progress(STAGES[stage], stage / len(STAGES), msg)

        def tick(name, t0):
            timings[name] = round(time.time() - t0, 2)

        # 1 ---- the artwork ------------------------------------------------------------------------
        t0 = time.time()
        say(0, "reading the artwork")
        prep = prepare_image(image, remove_bg=True, foreground_ratio=0.85, matting=o.matting,
                             max_side=o.reference_max_side)
        notes += list(prep.warnings)
        cutout = prep.cutout.convert("RGBA")
        ref_rgb = np.asarray(cutout.convert("RGB"), dtype=np.float32) / 255.0
        ref_alpha = np.asarray(cutout.getchannel("A"), dtype=np.float32) / 255.0
        ref_mask = ref_alpha > 0.5
        if ref_mask.mean() > 0.97:
            notes.append("the artwork has no separable background - registration and texture edges may suffer")
        small = cutout.copy()
        small.thumbnail((1024, 1024), Image.LANCZOS)
        tick("artwork", t0)

        # 2 ---- geometry -----------------------------------------------------------------------------
        t0 = time.time()
        if o.geometry == "auto" and not o.best_of and self._geometry is None:
            from .backends.geometry import AUTO_ORDER, make_geometry

            heavy_ok = False
            for nm in AUTO_ORDER:
                try:
                    if make_geometry(nm, self.device, weights=self.weights, engine=self.engine).status()[0]:
                        heavy_ok = True
                        break
                except Exception:
                    continue
            if not heavy_ok:
                # Never rank a flat 'relief' against a volume by front-silhouette IoU: a slab matches the front perfectly
                # by construction, so it always wins - that is how people ended up with a cut-out and no back or sides.
                o = replace(o, geometry="aura")
                notes.append("no neural shape model can run here, so the built-in volumetric model is used (real depth, lower "
                             "detail). For the full-quality result install Hunyuan3D-2 (runs in <= 8 GB): "
                             "aura-white setup --hunyuan   then   aura-white make <picture>")
        p = (lambda m: say(1, m))
        gkw = dict(seed=o.seed, progress=p, octree_resolution=cfg["octree"], steps=cfg["geo_steps"], reference=cutout)

        def _score(verts_, faces_) -> float:
            """Silhouette agreement of a candidate with the artwork (registered, like the real registration)."""
            n_, _c, _s = normalize_mesh(np.asarray(verts_, np.float32))
            r_ = register_reference(self.raster, n_, np.asarray(faces_, np.int64), ref_mask)
            if r_.iou < 0.7:
                R_, s_ = find_orientation(self.raster, n_, np.asarray(faces_, np.int64), ref_mask)
                if R_ is not None:
                    r_ = max(r_.iou, s_)
                    return float(r_)
            return float(r_.iou)

        if o.best_of:
            from .backends.geometry import make_geometry

            best = None
            cand_report = []
            for nm in o.best_of:
                try:
                    b_ = make_geometry(nm, self.device, weights=self.weights, engine=self.engine)
                    ok_, why_ = b_.status()
                    if not ok_:
                        notes.append(f"best_of: {nm} skipped ({why_})")
                        continue
                    say(1, f"geometry candidate: {nm}")
                    g_ = b_.generate(small, **gkw)
                    v_, f_, _ = keep_large_components(np.asarray(g_.verts, np.float32), np.asarray(g_.faces, np.int64),
                                                      None, o.min_component)
                    sc_ = _score(v_, f_)
                    cand_report.append((nm, round(sc_, 4)))
                    if best is None or sc_ > best[0]:
                        best = (sc_, b_, g_, v_, np.asarray(f_, np.int64))
                except Exception as e:
                    notes.append(f"best_of: {nm} failed ({type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''})")
                free_cuda()
            if best is None:
                raise BackendUnavailable("none of the best_of geometry back-ends could run: " + "; ".join(notes[-len(o.best_of):]))
            _sc, backend, geo, verts, faces = best
            notes.append("best_of silhouette scores: " + ", ".join(f"{n}={v}" for n, v in cand_report)
                         + f" -> using {getattr(backend, 'name', '?')}")
        else:
            backend = self._geometry_backend(o)
            say(1, f"geometry with {getattr(backend, 'name', type(backend).__name__)}")
            try:
                geo: GeometryResult = backend.generate(small, **gkw)
            except Exception as e:
                if self._geometry is not None or o.geometry != "auto" or isinstance(backend, AuraGeometry):
                    raise
                notes.append(f"{getattr(backend, 'name', 'geometry model')} failed ({type(e).__name__}: {e}); "
                             "fell back to the built-in Aura White geometry")
                free_cuda()
                backend = AuraGeometry(self.device, self.weights, engine=self.engine)
                geo = backend.generate(small, seed=o.seed, progress=p)
            verts, faces, _ = keep_large_components(np.asarray(geo.verts, np.float32),
                                                    np.asarray(geo.faces, np.int64), None, o.min_component)
            faces = np.asarray(faces, np.int64)
        tick("geometry", t0)

        # 3 ---- side views ---------------------------------------------------------------------------
        t0 = time.time()
        views = geo.views
        if views is None:
            mv = self._multiview_backend(o, int(cfg["mv_steps"]))
            if mv is not None:
                say(2, "imagining the other sides")
                try:
                    views = mv.generate(small, seed=o.seed, progress=lambda m: say(2, m))
                except BackendUnavailable:
                    raise
                except Exception as e:                    # never lose the whole run to the optional stage
                    notes.append(f"side views failed ({type(e).__name__}: {e}); hidden surfaces will be filled in")
                    views = None
                finally:
                    if hasattr(mv, "unload"):
                        mv.unload()
            else:
                notes.append("no side-view generator available: surfaces the artwork cannot see are filled from "
                             "the nearest visible colours")
        tick("side_views", t0)

        # 4 ---- orientation, registration ------------------------------------------------------------
        t0 = time.time()
        say(3, "aligning the artwork to the mesh")
        view_fg = [estimate_foreground(v) for v in views.images] if views is not None else []
        if views is not None and geo.absolute_scale:
            cams0 = [orbit_camera(a, e, 4.0, 30.0) for a, e in zip(ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS)]
            M, mirrored, best, ident = orientation_from_views(self.raster, verts, faces, view_fg, cams0)
            if not np.allclose(M, np.eye(3)):
                notes.append(f"mesh axes corrected using the generated views (IoU {ident:.2f} -> {best:.2f})")
                verts = (verts @ M.T).astype(np.float32)
                faces = faces[:, ::-1].copy() if mirrored else faces
        if signed_volume(verts, faces) < 0:                 # generators differ in winding; texturing needs outward normals
            faces = faces[:, ::-1].copy()
            notes.append("face winding was inverted and has been flipped")
        nv, centre, scale = normalize_mesh(verts)
        reg = register_reference(self.raster, nv, faces, ref_mask)
        if reg.iou < 0.70:
            R, s = find_orientation(self.raster, nv, faces, ref_mask)
            if R is not None and s > reg.iou + 0.05 and not np.allclose(R, np.eye(3)):
                notes.append(f"mesh orientation corrected by silhouette matching (IoU {reg.iou:.2f} -> {s:.2f})")
                nv = (nv @ R.T).astype(np.float32)          # a rotation about the origin keeps radius 1
                reg = register_reference(self.raster, nv, faces, ref_mask)
        notes += reg.notes
        if reg.iou < 0.80:
            notes.append(f"the artwork matches the mesh silhouette only loosely (IoU {reg.iou:.2f}); "
                         "texture details near edges may be off")
        tick("registration", t0)

        # 4b ---- topology repair, silhouette snap, residual completion ---------------------------------
        t0 = time.time()
        say(3, "repairing the mesh and completing missing parts")
        refine_report: Dict[str, object] = {}
        comp = None
        if o.repair and len(faces):
            try:
                from .mesh.repair import repair_mesh

                nv, faces, rep = repair_mesh(nv, faces)
                nv, faces = np.asarray(nv, np.float32), np.asarray(faces, np.int64)
                refine_report["repair"] = {k: v for k, v in rep.items() if k not in ("before", "after")}
                refine_report["topology_before"], refine_report["topology_after"] = rep["before"], rep["after"]
                if rep["before"]["boundary_edges"] or rep["before"]["nonmanifold_edges"]:
                    notes.append(f"topology repair: {rep['before']['boundary_edges']} boundary / "
                                 f"{rep['before']['nonmanifold_edges']} non-manifold edges -> "
                                 f"{rep['after']['boundary_edges']} / {rep['after']['nonmanifold_edges']}")
            except Exception as e:
                notes.append(f"topology repair skipped ({type(e).__name__}: {e})")
        if (o.snap or o.complete) and reg.iou >= 0.6 and len(faces):
            try:
                nv, comp, info = self._refine_to_artwork(nv, faces, reg, ref_rgb, ref_alpha, o, notes)
                refine_report.update(info)
            except Exception as e:                       # never lose the run to an optional refinement
                notes.append(f"silhouette refinement skipped ({type(e).__name__}: {e})")
        tick("refine", t0)

        # 4b(ii) -- render -> compare -> correct: if the mesh still doesn't match the artwork's outline well
        # enough, spend one more (higher-resolution, larger-step) snap pass on it rather than shipping as-is.
        if o.snap and o.refine_iterations > 0 and len(faces):
            t0 = time.time()
            try:
                from .forge import snap as fs

                cur_iou = refine_report.get("snap", {}).get("iou_after", reg.iou)
                extra = []
                for it in range(o.refine_iterations):
                    if cur_iou >= o.refine_target_iou:
                        break
                    say(3, f"render/compare pass {it + 1}: outline still {cur_iou:.3f}, refining further")
                    denser = min(int(o.working_res * (2 ** (it + 1))), ref_alpha.shape[0])
                    ref_a = np.asarray(Image.fromarray((ref_alpha * 255 + 0.5).astype(np.uint8))
                                       .resize((denser, denser), Image.BILINEAR), np.float32) / 255.0
                    res = fs.snap_to_silhouette(nv, faces, reg.camera, ref_a,
                                                max_shift_px=0.015 * denser, iterations=3)
                    extra.append(dict(pass_=it + 1, resolution=denser, iou_before=round(res.iou_before, 4),
                                      iou_after=round(res.iou_after, 4)))
                    if res.iou_after <= cur_iou + 1e-4:                    # no more progress to be had this way
                        break
                    nv, cur_iou = res.verts.astype(np.float32), res.iou_after
                if extra:
                    refine_report["render_compare_correct"] = extra
                    notes.append(f"render/compare/correct: outline IoU -> {cur_iou:.3f} after "
                                 f"{len(extra)} extra pass(es)")
            except Exception as e:
                notes.append(f"render/compare/correct pass skipped ({type(e).__name__}: {e})")
            tick("render_compare_correct", t0)

        locked_uv = None
        locked_report = None
        if o.fidelity == "source_locked":
            t0 = time.time()
            say(3, "building the source-locked 3D surface")
            from .fidelity import build_reference_locked_mesh

            # the generated geometry only supplies depth.  The builder already renders and compares at the image's
            # native resolution and adds vertices where pixels are still uncovered, so no coarse preflight/retry.
            locked = build_reference_locked_mesh(ref_alpha, nv, faces, reg.camera,
                                                 working_res=o.fidelity_resolution, point_step=o.fidelity_point_step,
                                                 boundary_step=o.fidelity_boundary_step, max_repair_passes=8)
            nv, faces, locked_uv = locked.verts, locked.faces, locked.uv
            locked_report = locked.report
            refine_report["source_locked"] = locked_report
            notes.append("source-locked fidelity mode: the supplied image is authoritative for the reference view")
            tick("source_locked", t0)

        # 5 ---- decimate + UV ------------------------------------------------------------------------
        t0 = time.time()
        say(4, "simplifying the mesh and unwrapping UVs")
        comp_faces = comp.n_faces if comp is not None else 0
        if o.fidelity == "source_locked":
            lo_v, lo_f, dec_backend = np.asarray(nv, np.float32), np.asarray(faces, np.int64), "source-locked"
        else:
            lo_v, lo_f, dec_backend = decimate(nv, faces, max(500, int(cfg["target_faces"]) - comp_faces))
        if comp is not None and comp.n_faces and o.fidelity != "source_locked":
            lo_f = np.concatenate([np.asarray(lo_f, np.int64), comp.faces + len(lo_v)], 0)
            lo_v = np.concatenate([np.asarray(lo_v, np.float32), comp.verts.astype(np.float32)], 0)
            faces = np.concatenate([faces, comp.faces + len(nv)], 0)
            nv = np.concatenate([nv, comp.verts.astype(np.float32)], 0)
        ln = vertex_normals(lo_v, lo_f)
        if o.fidelity == "source_locked":
            # UVs are the source image's own pixel coordinates by contract - no generic atlas here.
            uv_v, uv_f, uv = lo_v, lo_f, locked_uv[: len(lo_v)]
            vmap = np.arange(len(lo_v), dtype=np.int64)
            unwrap_backend = "source-image-locked"
        else:
            uv_v, uv_f, uv, vmap, unwrap_backend = unwrap(lo_v, lo_f, int(cfg["texture_size"]), o.unwrap)
        normals = ln[vmap]
        tick("uv", t0)

        # 6 ---- views -> texture ----------------------------------------------------------------------
        t0 = time.time()
        say(5, "baking the texture")
        gens: List[Tuple[np.ndarray, Camera]] = []
        gen_report = []
        n_views = 0 if views is None else len(views.images)
        if views is not None:
            az0 = 0.0 if geo.absolute_scale else reg.params["az"]
            for i, (img, fg) in enumerate(zip(views.images, view_fg)):
                a, e = ZERO123PP_AZIMUTHS[i], ZERO123PP_ELEVATIONS[i]
                if geo.absolute_scale:
                    cam0 = transform_camera(orbit_camera(a, e, 4.0, 30.0), centre, scale)
                else:
                    cam0 = orbit_camera(az0 + a, e, 4.0, 30.0)
                frac = float(fg.mean())
                if not 0.02 < frac < 0.8:                       # no plausible object on a flat background
                    gen_report.append(0.0)
                    continue
                al = align_view(self.raster, nv, faces, fg, cam0)
                gen_report.append(round(al.iou, 3))
                if al.iou >= 0.55:
                    gens.append((img, al.camera))
            if n_views - len(gens):
                notes.append(f"{n_views - len(gens)} generated view(s) did not match the mesh and were ignored")
        if o.photometric_refine and len(gens) >= 2 and reg.iou >= 0.5:
            pv, pf, _ = decimate(nv, faces, 12000)
            reg2 = refine_reference_photometric(self.raster, pv, pf, vertex_normals(pv, pf), ref_rgb, ref_alpha,
                                                gens, reg)
            if reg2 is not reg:
                notes += reg2.notes[len(reg.notes):]
                reg = reg2
        sources: List[ViewSource] = []
        if reg.iou >= 0.5:
            sources.append(ViewSource(ref_rgb, reg.camera, o.reference_weight, alpha=ref_alpha, name="artwork"))
        else:
            notes.append("artwork could not be registered; texture comes from the generated views only")
        for i, (img, cam) in enumerate(gens):
            sources.append(ViewSource(img, cam, o.generated_weight, name=f"side{i}", harmonize=True))
        want_mirror = o.back == "mirror" or (o.back == "auto" and geo.info.get("backend") == "relief")
        if want_mirror and reg.iou >= 0.5:
            rot = np.eye(4)
            rot[0, 0] = rot[1, 1] = -1.0                        # 180 degrees about the vertical (z) axis
            c2w = rot @ np.linalg.inv(np.asarray(reg.camera.w2c, np.float64))
            cam_back = Camera(np.linalg.inv(c2w), reg.camera.f, -reg.camera.cx, reg.camera.cy)
            sources.append(ViewSource(np.ascontiguousarray(ref_rgb[:, ::-1]), cam_back, 0.6 * o.reference_weight,
                                      alpha=np.ascontiguousarray(ref_alpha[:, ::-1]), name="back (mirrored artwork)"))
            notes.append("the back is textured with the mirrored artwork (depth-symmetric reconstruction)")
        if not sources:
            raise RuntimeError("no usable picture to texture the mesh with")
        if o.fidelity == "source_locked":
            from .fidelity import source_texture

            tex = source_texture(ref_rgb, ref_alpha, max_side=max(4096, int(cfg["texture_size"])),
                                 solid=0.98 if o.clean_rim else 0.5, defringe=o.clean_rim)

            class _LockedBake:
                texture = tex
                seen = np.ones(tex.shape[:2], np.float32)
                usage = {"artwork": 1.0}

            baked = _LockedBake()
        else:
            baked = bake_texture(self.raster, uv_v, uv_f, uv, normals, sources, size=int(cfg["texture_size"]),
                                 exponent=o.exponent, progress=lambda m: say(5, m))
        tick("texture", t0)

        # 7 ---- normal map ---------------------------------------------------------------------------
        t0 = time.time()
        nmap = None
        tangents = None
        if cfg["normal_map"]:
            say(6, "baking the normal map")
            if o.fidelity == "source_locked":
                # the source-locked shell IS the delivery surface; a flat tangent-space normal keeps the source
                # pixels from being reshaded, rather than inventing relief the picture never showed
                ns = min(int(cfg["texture_size"]), 2048)
                nmap = np.zeros((ns, ns, 3), np.uint8)
                nmap[..., 0], nmap[..., 1], nmap[..., 2] = 128, 128, 255
            elif len(faces) > 1.5 * len(lo_f):
                nmap = bake_normal_map(self.raster, nv, faces, uv_v, uv_f, uv, normals,
                                       size=min(int(cfg["texture_size"]), 2048), progress=lambda m: say(6, m))
                if nmap is None:
                    notes.append("normal map skipped (needs scipy: pip install scipy)")
            if nmap is not None:
                tangents = tangent_frames(uv_v, uv_f, uv, normals)
        tick("normal_map", t0)

        # 7b ---- ORM map (occlusion / roughness / metallic) -------------------------------------------
        # A flat map from the material's global roughness/metallic is what every prior version shipped
        # (or omitted entirely); it is still a placeholder for the occlusion channel specifically - a real
        # ambient-occlusion bake needs the high-poly surface and belongs to the Blender bake backend
        # (`aura_white.backends.blender_tools`) when that is available. Using it here, when it is, gives the
        # ORM map a real (not flat-255) occlusion channel at no extra cost.
        t0 = time.time()
        ts = int(cfg["texture_size"])
        orm = np.empty((ts, ts, 3), np.uint8)
        orm[..., 0] = 255
        orm[..., 1] = int(np.clip(round(o.roughness * 255), 0, 255))
        orm[..., 2] = int(np.clip(round(o.metallic * 255), 0, 255))
        ao_backend = "flat (no occlusion baked)"
        if cfg.get("blender_bake") and o.fidelity != "source_locked":
            try:
                from .backends.blender_tools import bake_ao_curvature

                ao = bake_ao_curvature(nv, faces, uv_v, uv_f, uv, size=ts, progress=lambda m: say(6, m))
                if ao is not None:
                    orm[..., 0] = ao
                    ao_backend = "Blender (Cycles)"
                    notes.append("ambient occlusion baked with Blender; ORM occlusion channel is real, not flat")
            except Exception as e:
                notes.append(f"Blender AO bake skipped ({type(e).__name__}: {e})")
        tick("orm", t0)

        # 8 ---- QA pictures --------------------------------------------------------------------------
        t0 = time.time()
        say(7, "rendering previews")
        previews: Dict[str, np.ndarray] = {}
        try:
            base_az = float(reg.params["az"])
            previews["preview"] = turntable_sheet(self.raster, uv_v, uv_f, uv, baked.texture,
                                                  tuple(base_az + a for a in (0, 60, 120, 180, 240, 300)),
                                                  elevation=float(reg.params["el"]), size=256, normals=normals)
            ov, m = render_textured(self.raster, uv_v, uv_f, uv, baked.texture, reg.camera, 512, bg=(0, 0, 0))
            ref_small = np.asarray(Image.fromarray((ref_rgb * 255).astype(np.uint8)).resize((512, 512)),
                                   dtype=np.float32) / 255.0
            am = np.asarray(Image.fromarray((ref_alpha * 255).astype(np.uint8)).resize((512, 512))) > 127
            over = ref_small * 0.55
            over[m & ~am] = (0.9, 0.2, 0.2)       # mesh sticks out of the artwork
            over[am & ~m] = (0.2, 0.3, 0.9)       # artwork not covered by the mesh
            previews["registration"] = np.concatenate([over, ov], axis=1)
            previews["difference"] = np.clip(np.abs(ov.astype(np.float32) - ref_small) * (am[..., None]), 0, 1)
            if views is not None and views.grid is not None:
                previews["side_views"] = views.grid
            previews["coverage"] = np.stack([baked.seen] * 3, -1)
        except Exception as e:                            # previews are a convenience, never fatal
            notes.append(f"preview rendering failed: {e}")
        tick("previews", t0)
        timings["total"] = round(time.time() - t_all, 2)

        # 9 ---- render -> compare: how well the FINAL textured, camera-matched render agrees with the source
        source_fidelity_iou = None
        source_fidelity_psnr = None
        try:
            from .evaluation import front_psnr, silhouette_iou

            size = min(2048 if o.fidelity == "source_locked" else 1024, ref_alpha.shape[0])
            source_fidelity_iou = silhouette_iou(uv_v, uv_f, reg.camera, ref_alpha, size=size)
            rendered, _rm = render_textured(self.raster, uv_v, uv_f, uv, baked.texture, reg.camera, size,
                                            bg=(0, 0, 0), light=0.0)
            rr = np.asarray(Image.fromarray((ref_rgb * 255 + 0.5).astype(np.uint8)).resize(
                (rendered.shape[1], rendered.shape[0]), Image.LANCZOS), np.float32) / 255.0
            aa = np.asarray(Image.fromarray((ref_alpha * 255).astype(np.uint8)).resize(
                (rendered.shape[1], rendered.shape[0]), Image.BILINEAR)) > 127
            if o.fidelity == "source_locked" and o.clean_rim:
                # --clean-rim deliberately alters the outline, so score the solid interior only
                from scipy.ndimage import binary_erosion

                aa = binary_erosion(aa, iterations=2) if aa.sum() > 64 else aa
            source_fidelity_psnr = front_psnr(rendered, rr, aa)
        except Exception as e:
            notes.append(f"source fidelity QA failed: {type(e).__name__}: {e}")
        target = o.fidelity_target_iou if o.fidelity == "source_locked" else o.refine_target_iou
        target_met = source_fidelity_iou is not None and source_fidelity_iou >= target
        if not target_met:
            msg = (f"source-view fidelity target not reached: silhouette IoU "
                  f"{'unmeasured' if source_fidelity_iou is None else f'{source_fidelity_iou:.4f}'} < {target:.3f}")
            if o.fidelity == "source_locked" and o.strict_fidelity:
                raise RuntimeError(msg + " (strict_fidelity=True: refusing to publish; set strict_fidelity=False "
                                         "to ship it anyway with this noted, or lower fidelity_target_iou)")
            notes.append(msg + " - shipped anyway; see asset_difference.png for exactly where it disagrees")

        report = dict(
            geometry=getattr(backend, "name", type(backend).__name__), geometry_info=geo.info,
            side_views=views is not None, faces_high=int(len(faces)), faces_final=int(len(uv_f)),
            vertices_final=int(len(uv_v)), texture_size=int(max(baked.texture.shape[:2])), preset=o.quality,
            decimation=dec_backend, unwrap=unwrap_backend, raster=self.raster.backend,
            reference_iou=round(float(reg.iou), 4), reference_camera={k: round(float(v), 4) for k, v in reg.params.items()},
            source_fidelity_iou=None if source_fidelity_iou is None else round(float(source_fidelity_iou), 5),
            source_fidelity_psnr_db=None if source_fidelity_psnr is None else round(float(source_fidelity_psnr), 3),
            source_fidelity_target=target, source_fidelity_target_met=bool(target_met),
            fidelity_mode=o.fidelity, reference_view_is_authoritative=(o.fidelity == "source_locked"),
            side_view_iou=gen_report, view_usage={k: round(v, 3) for k, v in baked.usage.items()},
            normal_map=nmap is not None, orm=True, ao_backend=ao_backend, unlit=bool(o.fidelity == "source_locked"),
            timings=timings, notes=notes, device=str(self.raster.device), refine=refine_report)

        # 10 ---- LODs + collision (optional) ----------------------------------------------------------
        lods: List[object] = []
        collision = None
        n_lods = int(cfg.get("lods") or 0)
        if n_lods > 0:
            t0 = time.time()
            say(7, f"generating {n_lods} LOD level(s)")
            try:
                from .mesh.lod import generate_lods

                fracs = [1.0] + [max(0.03, 0.35 ** k) for k in range(1, n_lods + 1)]
                lods = generate_lods(uv_v, uv_f, levels=tuple(fracs[: n_lods + 1]))
                report["lods"] = [dict(name=l.name, faces=l.faces_actual, watertight=l.watertight) for l in lods]
            except Exception as e:
                notes.append(f"LOD generation skipped ({type(e).__name__}: {e})")
            tick("lods", t0)
        if o.collision:
            t0 = time.time()
            say(7, "generating a collision mesh")
            try:
                from .mesh.lod import convex_hull_collision

                collision = convex_hull_collision(uv_v)
                report["collision"] = dict(kind=collision.kind, faces=collision.faces_count,
                                           volume=round(collision.volume, 6))
            except Exception as e:
                notes.append(f"collision mesh skipped ({type(e).__name__}: {e})")
            tick("collision", t0)
        report["notes"] = notes
        report["timings"] = timings

        return TexturedResult(uv_v, uv_f, uv, normals, tangents, baked.texture, nmap, orm,
                              bool(o.fidelity == "source_locked"), (centre, float(scale)), report, previews,
                              o.roughness, o.metallic, lods, collision)

    # -- multi-view entry point -------------------------------------------------------------------------
    _AZIMUTH_GUESS = {"front": 0.0, "3-4-front": -35.0, "3-4 front": -35.0, "right": 90.0, "right_side": 90.0,
                      "side": 90.0, "back": 180.0, "rear": 180.0, "left": 270.0, "left_side": 270.0,
                      "3-4-back": 145.0, "3-4 back": 145.0}

    def generate_multiview(self, views: Dict[str, object], azimuths: Optional[Dict[str, float]] = None,
                           elevations: Optional[Dict[str, float]] = None, primary: Optional[str] = None,
                           progress: Progress = None, **overrides) -> "TexturedResult":
        """Reconstruct from several pictures of the SAME object (front/side/back, or a turntable subset) jointly,
        instead of guessing the hidden side from one image. `views` maps a name to an image (path or PIL Image);
        `azimuths` gives each view's camera angle in degrees when the name isn't one of front/back/left/right/
        3-4-front/3-4-back (recognised automatically) - an unlabelled, unguessed view is spread evenly around the
        turntable. `elevations` defaults every view to the same 10 degrees: this does not solve for a different
        camera height per shot, so pictures taken from noticeably different heights will register worse than a
        matched turntable rig. The primary view (default: 'front' if present, else the first given) drives geometry
        generation and residual completion (chains/cords/patches); every view feeds the joint silhouette snap and
        the texture bake.

        Returns the same `TexturedResult` as `generate()`; `--best-of`, LODs and collision are not part of this
        entry point yet - post-process the output with `aura-white lod` / `aura-white repair` if you need them."""
        from .image.preprocess import prepare_image
        from .texture import bake_texture, decimate, find_orientation, register_reference, unwrap, ViewSource
        from .texture.preview import turntable_sheet

        if len(views) < 2:
            raise ValueError("generate_multiview needs at least 2 views; use generate() for a single image")
        o = replace(self.options, **overrides) if overrides else self.options
        on_gpu = self.raster.device.type == "cuda"
        cfg = o.resolved(on_gpu)
        say = (lambda stage, msg: progress(stage, 0.0, msg)) if progress else (lambda *_: None)
        notes: List[str] = []
        t_all = time.time()

        names = list(views.keys())
        primary_name = primary or ("front" if "front" in views else names[0])
        az = {n: (azimuths or {}).get(n, self._AZIMUTH_GUESS.get(n.lower())) for n in names}
        unresolved = [n for n in names if az[n] is None and n != primary_name]
        for i, n in enumerate(unresolved):                # spread anything unrecognised evenly, skipping 0
            az[n] = 360.0 * (i + 1) / (len(unresolved) + 1)
        az[primary_name] = az.get(primary_name) or 0.0
        el = {n: (elevations or {}).get(n, 10.0) for n in names}

        say("prepare", f"reading {len(names)} views")
        prepped = {}
        for n, img in views.items():
            p = prepare_image(img, remove_bg=True, foreground_ratio=0.85, matting=o.matting,
                              max_side=o.reference_max_side)
            prepped[n] = p
        primary_p = prepped[primary_name]
        small = primary_p.cutout.resize((min(768, primary_p.cutout.width),) * 2) \
            if primary_p.cutout.width > 768 else primary_p.cutout

        say("geometry", f"geometry from the '{primary_name}' view")
        backend = self._geometry_backend(o)
        geo = backend.generate(small, seed=o.seed, progress=lambda m: say("geometry", m), reference=primary_p.cutout)
        verts, faces, _ = keep_large_components(np.asarray(geo.verts, np.float32), np.asarray(geo.faces, np.int64),
                                                None, o.min_component)
        faces = np.asarray(faces, np.int64)
        nv, _c, _s = normalize_mesh(verts)

        say("registration", f"registering the '{primary_name}' view")
        ref_alpha0 = np.asarray(primary_p.cutout.getchannel("A"), np.float32) / 255.0
        try:
            R0, s0 = find_orientation(self.raster, nv, faces, ref_alpha0)
            if R0 is not None and s0 > 0:
                nv, faces = nv @ R0.T, (faces if np.linalg.det(R0) > 0 else faces[:, ::-1])
        except Exception as e:
            notes.append(f"orientation search skipped: {type(e).__name__}: {e}")
        reg0 = register_reference(self.raster, nv, faces, ref_alpha0, prior_az=0.0)
        cams = {primary_name: reg0.camera}
        alphas = {primary_name: ref_alpha0}
        rgbs = {primary_name: np.asarray(primary_p.cutout.convert("RGB"), np.float32) / 255.0}
        view_iou = {primary_name: round(float(reg0.iou), 4)}

        for n in names:
            if n == primary_name:
                continue
            p = prepped[n]
            a = np.asarray(p.cutout.getchannel("A"), np.float32) / 255.0
            reg = register_reference(self.raster, nv, faces, a, prior_az=float(az[n]), prior_el=float(el[n]),
                                     az_range=30.0, allow_flip=False)
            cams[n], alphas[n] = reg.camera, a
            rgbs[n] = np.asarray(p.cutout.convert("RGB"), np.float32) / 255.0
            view_iou[n] = round(float(reg.iou), 4)
            if reg.iou < 0.5:
                notes.append(f"view '{n}': low registration confidence (IoU {reg.iou:.2f}) at azimuth "
                            f"{az[n]:.0f} deg - check the azimuth hint or the picture itself")
        say("registration", "joint silhouette snap across all views")
        try:
            from .forge import snap as fs

            res = fs.snap_to_silhouettes(nv, faces, [(cams[n], alphas[n]) for n in names], iterations=3)
            nv = res.verts.astype(np.float32)
            notes.append(f"joint snap: mean outline IoU across {len(names)} views "
                        f"{res.iou_before:.3f} -> {res.iou_after:.3f}")
        except Exception as e:
            notes.append(f"joint silhouette snap skipped: {type(e).__name__}: {e}")

        comp = None
        if o.complete:
            try:
                nv, comp, _info = self._refine_to_artwork(nv, faces, reg0, rgbs[primary_name], alphas[primary_name],
                                                           replace(o, snap=False), notes)
            except Exception as e:
                notes.append(f"residual completion (primary view only) skipped: {type(e).__name__}: {e}")

        if o.repair and len(faces):
            try:
                from .mesh.repair import repair_mesh

                nv, faces, rep = repair_mesh(nv, faces)
                notes.append(f"topology repair: watertight={rep['watertight']}")
            except Exception as e:
                notes.append(f"topology repair skipped: {type(e).__name__}: {e}")

        say("uv", "simplifying and unwrapping")
        comp_faces = comp.n_faces if comp is not None else 0
        lo_v, lo_f, dec_backend = decimate(nv, faces, max(500, int(cfg["target_faces"]) - comp_faces))
        if comp is not None and comp.n_faces:
            lo_f = np.concatenate([lo_f, comp.faces + len(lo_v)], 0)
            lo_v = np.concatenate([lo_v, comp.verts.astype(np.float32)], 0)
        ln = vertex_normals(lo_v, lo_f)
        uv_v, uv_f, uv, vmap, unwrap_backend = unwrap(lo_v, lo_f, int(cfg["texture_size"]), o.unwrap)
        normals = ln[vmap]

        say("texture", f"baking from {len(names)} views")
        sources = [ViewSource(rgbs[n], cams[n], o.reference_weight, alpha=alphas[n], name=n, harmonize=(n != primary_name))
                  for n in names]
        baked = bake_texture(self.raster, uv_v, uv_f, uv, normals, sources, size=int(cfg["texture_size"]),
                             exponent=o.exponent, progress=lambda m: say("texture", m))

        nmap, tangents = None, None
        if cfg["normal_map"] and len(faces) > 1.5 * len(lo_f):
            from .texture import bake_normal_map, tangent_frames

            nmap = bake_normal_map(self.raster, nv, faces, uv_v, uv_f, uv, normals,
                                   size=min(int(cfg["texture_size"]), 2048), progress=lambda m: say("normal", m))
            if nmap is not None:
                tangents = tangent_frames(uv_v, uv_f, uv, normals)

        ts = int(cfg["texture_size"])
        orm = np.empty((ts, ts, 3), np.uint8)
        orm[..., 0] = 255
        orm[..., 1] = int(np.clip(round(o.roughness * 255), 0, 255))
        orm[..., 2] = int(np.clip(round(o.metallic * 255), 0, 255))

        previews: Dict[str, np.ndarray] = {}
        try:
            previews["preview"] = turntable_sheet(self.raster, uv_v, uv_f, uv, baked.texture,
                                                  tuple(range(0, 360, 60)), elevation=15.0, size=256, normals=normals)
        except Exception as e:
            notes.append(f"preview rendering failed: {e}")

        report = dict(geometry=getattr(backend, "name", type(backend).__name__), geometry_info=geo.info,
                      mode="multiview", views=names, view_azimuths={n: round(float(az[n]), 1) for n in names},
                      view_iou=view_iou, faces_high=int(len(faces)), faces_final=int(len(uv_f)),
                      vertices_final=int(len(uv_v)), texture_size=ts, decimation=dec_backend,
                      unwrap=unwrap_backend, view_usage={k: round(v, 3) for k, v in baked.usage.items()},
                      normal_map=nmap is not None, orm=True, notes=notes, device=str(self.raster.device),
                      timings={"total": round(time.time() - t_all, 2)})
        return TexturedResult(uv_v, uv_f, uv, normals, tangents, baked.texture, nmap, orm, False,
                              (np.zeros(3), 1.0), report, previews, o.roughness, o.metallic)
