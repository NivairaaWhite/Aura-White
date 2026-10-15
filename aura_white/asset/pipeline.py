"""Picture -> game-ready asset pack.

    build_asset("castle.png", "out/castle")      # from Python
    aura-white asset castle.png -o out/castle    # from a shell

What comes out (see ``report.md`` / ``game.json`` in the pack):  a textured GLB in metres (Y up, +Z = the side the picture
shows, pivot where you want it), a LOD chain sharing one atlas, a convex-hull collision set with engine naming,
albedo / normal / ORM (/ emissive) maps, an unlit variant, an OBJ for tools that prefer it, a validation report and
preview sheets.  The reference view is reproduced exactly (the picture is the texture), the sides are inferred - the
report says how.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from .. import __version__
from . import atlas as A
from . import collision as C
from . import maps as M
from . import render as R
from . import validate as V
from .gltf import write_glb
from .layers import assemble, merge_layers
from .pipeline_core import bake_atlas
from .meshlock import MeshLockBuilder
from .proxies import ArenaBuilder, IslandBuilder, ReliefBuilder, SphereBuilder
from .subject import load_subject

BUILDERS = {"sphere": SphereBuilder, "character": ReliefBuilder, "prop": ReliefBuilder, "building": ReliefBuilder,
            "arena": ArenaBuilder, "island": IslandBuilder}
# size of a new asset in metres: (which dimension, value) - override with size= / height=
DEFAULT_SIZE = {"character": ("height", 1.7), "prop": ("longest", 1.8), "building": ("height", 20.0),
                "arena": ("longest", 40.0), "island": ("longest", 80.0), "sphere": ("longest", 2.0)}
DEFAULT_PIVOT = {"character": "base", "prop": "base", "building": "base", "arena": "base", "island": "center",
                 "sphere": "center"}
ENGINES = ("generic", "godot", "unity", "unreal", "ursina")


@dataclass
class AssetOptions:
    kind: str = "auto"                   # auto | sphere | character | prop | building | arena | island
    name: Optional[str] = None
    matte: str = "auto"                  # auto | alpha | plain | disc | rembg | border | file
    matte_file: Optional[str] = None     # a black/white (or alpha) PNG for the cut-out
    rembg_model: str = "u2net"
    elev: Optional[float] = None         # camera elevation of the artwork in degrees (arena/island: estimated / 28)
    size: Optional[float] = None         # metres, longest edge of the bounding box
    height: Optional[float] = None       # metres, height (wins over size)
    pivot: Optional[str] = None          # base | center | bottom
    engine: str = "generic"              # naming of collision nodes inside the main GLB
    lods: int = 4
    atlas: Optional[int] = None          # max atlas edge in texels (default 4096 for arena/island, else 2048)
    square: bool = False                 # pad the atlas to a square (old mobile GPUs)
    max_faces: int = 40000               # triangle budget of LOD0 for relief meshes
    back: str = "auto"                   # auto | soft | mirror | flat | echo  (the unseen back; auto: echo for props, buildings, arenas, islands under mesh-lock, soft otherwise)
    back_image: Optional[str] = None     # a picture of the object seen from behind (same framing, any background)
    extra_views: Optional[list] = None   # gen3d.paint.ExtraView pictures (back / sides / synthesised) projected into the atlas
    generate: str = "never"              # never | auto | always: with no proxy, make a REAL 3-D mesh with the gen3d engine
                                         # (auto: for character / prop / building; the CLI turns this on).  Library default
                                         # stays "never" so existing callers keep the old behaviour.
    generate_opts: Optional[dict] = None # gen3d options: quality, tier, allow_basic, seed, device, flat_ok ...
    proxy: Optional[str] = None          # mesh-lock: a GLB / OBJ / PLY made by any generator; its shape is used, locked to the picture
    proxy_yaw: Optional[float] = None    # mesh-lock: turn of the supplied mesh about +Y in degrees (default: found automatically)
    hi_normals: bool = True              # mesh-lock: bake the full-resolution mesh's detail into the normal / occlusion maps
    detail: Optional[float] = None       # mesh-lock: how much picture detail is carried onto the unseen sides (0..1)
    proxy_colour: bool = True            # mesh-lock: colour the sides / back with the supplied mesh's own texture when it has one
    normal_strength: Optional[float] = None
    roughness: Optional[float] = None
    metallic: Optional[float] = None
    emissive: str = "auto"               # auto | off
    collision: bool = True
    obj: bool = True
    preview: bool = True
    max_side: int = 2048
    extra: Dict[str, object] = field(default_factory=dict)   # builder options (wall_ratio, floor_ratio, rcap, ...)


def slug(text: str, fallback: str = "asset") -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not s or re.fullmatch(r"(file_)?[0-9a-f_]{16,}", s) or re.fullmatch(r"[0-9_]+", s):
        return fallback
    return s[:48]


def _atlas_for(kind: str, opts: AssetOptions) -> int:
    return int(opts.atlas or (4096 if kind in ("arena", "island") else 2048))


def _winding_error(cam, v, f) -> float:
    """Share of the pixels the reference camera sees whose triangle faces away from it (flipped winding)."""
    from .camera import rasterize_ortho
    sx, sy, t = cam.project(v)
    tid, _, _ = rasterize_ortho(np.stack([sx, sy], 1), t, f, cam.height, cam.width, nearest_is_larger=True)
    hit = tid >= 0
    if not hit.any():
        return 0.0
    tri = f[tid[hit]]
    n = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return float(((n @ cam.to_camera) < -1e-6).mean())


def _collision_names(engine: str, name: str, n: int) -> List[str]:
    if engine == "godot":
        return [f"{name}_col{i + 1:02d}-convcolonly" for i in range(n)]
    if engine == "unreal":
        return [f"UCX_{name}_{i:02d}" for i in range(n)]
    return [f"{name}_Collider_{i + 1:02d}" for i in range(n)]


def _write_obj_multi(path: Path, hulls, names):
    lines, off = [f"# collision hulls ({len(hulls)})"], 0
    for (v, f, _), nm in zip(hulls, names):
        lines.append(f"o {nm}")
        lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v]
        lines += [f"f {a + 1 + off} {b + 1 + off} {c + 1 + off}" for a, b, c in f]
        off += len(v)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_asset(src, out_dir, opts: Optional[AssetOptions] = None, log: Callable[[str], None] = print) -> dict:
    t0 = time.time()
    opts = opts or AssetOptions()
    if opts.engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    out = Path(out_dir)
    for sub in ("textures", "lods", "collision", "preview", "obj"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. understand the picture
    s = load_subject(src, matte=opts.matte, matte_file=opts.matte_file, kind=opts.kind, max_side=opts.max_side,
                     name=opts.name, rembg_model=opts.rembg_model)
    kind = s.kind
    name = slug(opts.name or s.name, fallback=kind)
    log(f"[1/6] {name}: {kind} ({s.kind_reason}); matte = {s.matte}")
    if (not opts.proxy and opts.generate != "never"
            and (opts.generate == "always" or kind in ("character", "prop", "building"))):
        # A picture alone says nothing about its back and sides.  Instead of cutting the picture out as a slab (front =
        # picture, back = its mirror) a true 3-D generator builds the volume; mesh-lock then pins it to the picture.
        from ..gen3d.engine import proxy_for_subject
        pv, pf, gviews, gmeta = proxy_for_subject(s, kind, opts.generate_opts, log)
        opts = replace(opts, proxy=(pv, pf), proxy_yaw=0.0, extra_views=list(opts.extra_views or []) + list(gviews))
        log(f"      generated a 3-D mesh with {gmeta['backend']} [{gmeta['tier']}]: depth {gmeta['depth_ratio']:.0%} of width, "
            f"outline agreement {gmeta['view_iou']:.3f} ({gmeta['reason']})")
    Builder = MeshLockBuilder if opts.proxy else BUILDERS[kind]
    bopts = dict(opts.extra)
    if opts.elev is not None:
        bopts["elev"] = opts.elev
    bopts["max_faces"] = opts.max_faces
    bopts["lods"] = opts.lods
    back_eff = opts.back
    if back_eff == "auto":
        back_eff = "echo" if (opts.proxy and kind in ("prop", "building", "arena", "island")) else "soft"
    if back_eff == "echo" and not opts.proxy:
        back_eff = "soft"                              # only the mesh-lock builder knows the echo fill
    bopts["back"] = back_eff
    bopts["atlas_cap"] = _atlas_for(kind, opts)
    if opts.proxy:
        bopts["proxy"] = opts.proxy
        bopts["proxy_yaw"] = opts.proxy_yaw
        bopts["proxy_colour"] = opts.proxy_colour
        if opts.detail is not None:
            bopts["detail"] = opts.detail
        log(f"      mesh-lock: fitting {Path(str(opts.proxy)).name if isinstance(opts.proxy, (str, Path)) else 'the generated mesh'} to the picture ...")
    cam = Builder.camera(s, bopts)
    b = Builder(s, cam, bopts)

    # ------------------------------------------------------------------ 2. geometry for every level of detail
    n_lod = int(max(1, min(opts.lods, 4)))
    layers_by_lod = [b.layers(k) for k in range(n_lod)]
    log("[2/6] geometry: " + ", ".join(f"LOD{k} {sum(len(L.faces) for L in ls):,} tris" for k, ls in enumerate(layers_by_lod)))

    # ------------------------------------------------------------------ 3. texture atlas (one for every LOD)
    back_img = None
    if opts.back_image:
        bi = Image.open(opts.back_image).convert("RGB").resize(s.size, Image.Resampling.LANCZOS)
        back_img = A.fill_outside(np.asarray(bi)[:, ::-1].copy(), s.alpha)   # seen from behind -> registered with the front
    atlas_cap = _atlas_for(kind, opts)
    if hasattr(b, "set_back"):
        b.set_back(back_img, back_eff)
    tex, (aw, ah) = bake_atlas(b, layers_by_lod[0], max_size=atlas_cap, back_mode=back_eff if back_eff != "echo" else "soft",
                               back_image=back_img)
    if opts.square and aw != ah:
        side = max(aw, ah)
        sq = np.zeros((side, side, 3), np.uint8)
        sq[:ah, :aw] = tex
        # the charts keep their pixel positions: only the uv divisor changes
        tex, aw, ah = sq, side, side
    log(f"[3/6] atlas {aw}x{ah}")
    meshes = []
    for ls in layers_by_lod:
        v, f, uv, n, _ = assemble(merge_layers(ls), b.charts, (aw, ah))
        keep = ~((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))
        meshes.append((v.astype(np.float64), f[keep], uv, n))

    if opts.extra_views:
        from ..gen3d.paint import project_views_into_atlas
        tex, prep = project_views_into_atlas(tex, (meshes[0][0], meshes[0][1], meshes[0][2]), opts.extra_views,
                                             ref_project=cam.project, ref_size_hw=(cam.height, cam.width), log=log)
        log(f"      extra views: {prep['updated_fraction']:.0%} of the atlas re-coloured from {len(prep['views'])} view(s)")

    # ------------------------------------------------------------------ 4. units, pivot, maps, collision
    v0 = meshes[0][0]
    lo, hi = v0.min(0), v0.max(0)
    ext_px = hi - lo
    mode, val = DEFAULT_SIZE[kind]
    if opts.height:
        scale = opts.height / ext_px[1]
    elif opts.size:
        scale = opts.size / float(ext_px.max())
    else:
        scale = val / float(ext_px[1] if mode == "height" else ext_px.max())
    pivot_kind = opts.pivot or DEFAULT_PIVOT[kind]
    ctr = 0.5 * (lo + hi)
    pivot = np.array([ctr[0], lo[1] if pivot_kind in ("base", "bottom") else ctr[1], ctr[2]])
    to_m = lambda p: ((np.asarray(p, np.float64) - pivot) * scale)
    game = [(to_m(v).astype(np.float32), f, uv, n) for v, f, uv, n in meshes]
    size_m = (ext_px * scale).tolist()

    rough, metal, nstr, aoo = M.KIND_MATERIAL[kind]
    rough = opts.roughness if opts.roughness is not None else rough
    metal = opts.metallic if opts.metallic is not None else metal
    nstr = opts.normal_strength if opts.normal_strength is not None else nstr
    images = {"albedo": tex}
    cavity = None
    hi = getattr(b, "hi_mesh", None)
    if hi is not None and opts.hi_normals:
        from ..texture.normals import tangent_frames as _tf
        from .hibake import bake_hi_maps
        mv, mf, muv, mn = meshes[0]
        log("      baking the full-resolution mesh's detail into the normal map ...")
        nmap, cavity = bake_hi_maps(hi[0], hi[1], mv, mf, muv, mn, _tf(mv, mf, muv.astype(np.float64), mn), (aw, ah),
                                    strength=float(opts.normal_strength if opts.normal_strength is not None else 1.0))
        images["normal"] = nmap
    elif nstr > 0:
        images["normal"] = M.normal_from_albedo(tex, nstr)
    images["orm"] = M.orm_map(tex, rough, metal, aoo, cavity=cavity)
    if opts.emissive == "auto" and kind != "sphere":
        em = M.emissive_map(tex)
        if em is not None:
            images["emissive"] = em
    from ..texture.normals import tangent_frames
    tangents = []
    for v, f, uv, n in game:
        tangents.append(tangent_frames(v, f, uv.astype(np.float64), n) if "normal" in images else None)

    hulls, hull_names = [], []
    if opts.collision:
        groups = b.collision_groups(layers_by_lod[0])
        hulls = [(to_m(hv).astype(np.float32), hf, vol * scale ** 3)
                 for hv, hf, vol in C.build_hulls([g for g in groups])]
        hull_names = _collision_names(opts.engine, name, len(hulls))
    log(f"[4/6] {size_m[0]:.2f} x {size_m[1]:.2f} x {size_m[2]:.2f} m, pivot {pivot_kind}; maps: {', '.join(images)}; "
        f"{len(hulls)} collision hulls")

    # ------------------------------------------------------------------ 5. files
    for k, img in images.items():
        Image.fromarray(img).save(out / "textures" / f"{name}_{k}.png")
    werr = _winding_error(cam, meshes[0][0], meshes[0][1])
    double_sided = kind in ("arena", "island", "building") or werr > 0.01
    mat_lit = dict(name=f"{name}_mat", base="albedo", normal="normal" if "normal" in images else None, orm="orm",
                   emissive="emissive" if "emissive" in images else None, double_sided=double_sided, roughness=rough,
                   metallic=metal)
    mat_unlit = dict(name=f"{name}_unlit", base="albedo", unlit=True, double_sided=double_sided)

    def node(k, nm=None, extras=None):
        v, f, uv, n = game[k]
        return dict(name=nm or (name if k == 0 else f"{name}_LOD{k}"), verts=v, faces=f, uv=uv, normals=n,
                    tangents=tangents[k], material=0, extras=extras or {})

    main = [node(0, extras={"lod": 0, "triangles": int(len(game[0][1])), "tier": b.tier})]
    if opts.engine in ("godot", "unreal"):
        main += [dict(name=nm, verts=hv, faces=hf) for (hv, hf, _), nm in zip(hulls, hull_names)]
    gen = f"Aura White {__version__}"
    files = {}
    write_glb(out / f"{name}.glb", main, [mat_lit], images, generator=gen)
    files["glb"] = f"{name}.glb"
    write_glb(out / f"{name}_unlit.glb", [node(0)], [mat_unlit], {"albedo": tex}, generator=gen)
    files["glb_unlit"] = f"{name}_unlit.glb"
    if n_lod > 1:
        lods = [node(k, f"{name}_LOD{k}", {"lod": k, "triangles": int(len(game[k][1]))}) for k in range(n_lod)]
        write_glb(out / "lods" / f"{name}_lods.glb", lods, [mat_lit], images, lod_group=list(range(n_lod)), generator=gen)
        files["glb_lods"] = f"lods/{name}_lods.glb"
    if hulls:
        cn = _collision_names(opts.engine, name, len(hulls))
        write_glb(out / "collision" / f"{name}_collision.glb", [dict(name=nm, verts=hv, faces=hf) for (hv, hf, _), nm in zip(hulls, cn)],
                  [], {}, generator=gen)
        _write_obj_multi(out / "collision" / f"{name}_collision.obj", hulls, cn)
        files["collision_glb"] = f"collision/{name}_collision.glb"
        files["collision_obj"] = f"collision/{name}_collision.obj"
    if opts.obj:
        from ..mesh.textured import write_obj_textured
        v, f, uv, n = game[0]
        write_obj_textured(out / "obj" / f"{name}.obj", v, f, uv, tex, n, images.get("normal"), name=name)
        files["obj"] = f"obj/{name}.obj"
    files["textures"] = {k: f"textures/{name}_{k}.png" for k in images}

    # ------------------------------------------------------------------ 6. validation and previews
    lod_info = []
    for k, (v, f, uv, n) in enumerate(game):
        lod_info.append(dict(name=f"LOD{k}", triangles=int(len(f)), vertices=int(len(v))))
    est = V.edge_stats(game[0][0], game[0][1], tol=max(1e-4, 1e-4 * float(np.max(size_m))))
    rgb_ref, hit, _ = R.render_reference(cam, meshes[0][0], meshes[0][1], meshes[0][2], tex)
    fid = R.fidelity_report(s.rgb, s.alpha, rgb_ref, hit)
    iou, psnr = fid["iou"], fid["psnr"]
    uvr = V.uv_report(game[0][2], game[0][1], (min(aw, 2048), min(ah, 2048)))
    m_per_px = float(scale)
    dens = {f"chart{c.cid}_{c.kind}": round(V.texel_density(c.scale, m_per_px), 1) for c in b.charts.values()}
    ok_pot = all((x & (x - 1)) == 0 for x in (aw, ah))
    checks = [
        ("reference view matches the picture (IoU >= 0.95, or outline within 2 px for 98 % of it)",
         iou >= 0.95 or fid["outline_err"] <= 0.02,
         f"IoU {iou:.4f}, outline error {100 * fid['outline_err']:.2f} %, PSNR {min(psnr, 99):.1f} dB, away from the outline {fid['psnr_core']:.1f} dB"),
        ("atlas is a power of two", ok_pot, f"{aw}x{ah}"),
        ("all UVs inside 0..1", bool(uvr["in_bounds"]), f"{uvr['min']} .. {uvr['max']}"),
        ("no degenerate triangles", V.degenerate_faces(*game[0][:2]) == 0, f"{V.degenerate_faces(*game[0][:2])}"),
        ("face winding faces the reference camera", werr < 0.02, f"{100 * werr:.2f} % flipped"),
        ("every LOD is smaller than the one before", all(lod_info[i + 1]["triangles"] < lod_info[i]["triangles"] for i in range(len(lod_info) - 1)),
         ", ".join(str(l["triangles"]) for l in lod_info)),
        ("collision is convex hulls only", True if hulls else None, f"{len(hulls)} hulls, {sum(len(h[1]) for h in hulls)} triangles"),
    ]
    manifest = {
        "aura_white": __version__, "schema": 1, "name": name, "source": str(src) if isinstance(src, (str, Path)) else "array",
        "kind": kind, "kind_reason": s.kind_reason, "tier": b.tier, "matte": s.matte,
        "units": "metre", "up_axis": "+Y", "front_axis": "+Z",
        "size_m": [round(x, 4) for x in size_m], "pivot": {"type": pivot_kind}, "metres_per_source_pixel": scale,
        "atlas": {"width": aw, "height": ah, "texels_per_metre_at_reference_view": dens, "uv_overlap_factor": round(float(uvr["overlap_factor"]), 3)},
        "lods": lod_info, "materials": {"roughness": rough, "metallic": metal, "double_sided": double_sided,
                                        "maps": list(images)},
        "collision": {"hulls": len(hulls), "node_names": hull_names or None, "engine_naming": opts.engine,
                      "triangles": int(sum(len(h[1]) for h in hulls))},
        "topology_lod0": est, "fidelity_reference_view": {"iou": round(iou, 5), "psnr_db": round(min(psnr, 99.0), 2),
                                    "outline_error": round(fid["outline_err"], 5), "psnr_core_db": round(fid["psnr_core"], 2)},
        "checks": [{"check": c, "ok": ok, "detail": d} for c, ok, d in checks],
        "info": {k: v for k, v in b.info.items()}, "notes": list(s.notes) + list(b.notes), "files": files,
        "seconds": round(time.time() - t0, 1),
    }
    if opts.preview:
        cams = dict(elevation=float(cam.elev) if cam.elev > 1 else 8.0)
        v, f, uv, n = meshes[0]
        sheet = R.turntable(v, f, uv, n, tex, azimuths=(0, 45, 90, 135, 180, 225, 270, 315), tile=320, **cams)
        Image.fromarray(sheet).save(out / "preview" / "turntable.png")
        tiles = [R.turntable(mv, mf, muv, mn, tex, azimuths=(35,), tile=360, labels=False, **cams) for mv, mf, muv, mn in meshes]
        Image.fromarray(np.concatenate(tiles, 1)).save(out / "preview" / "lods.png")
        diff = np.clip(np.abs(rgb_ref.astype(np.int16) - s.rgb.astype(np.int16)).mean(2) * 4, 0, 255).astype(np.uint8)
        cmp_ = np.concatenate([s.rgb, rgb_ref, np.repeat(diff[..., None], 3, 2)], 1)
        im = Image.fromarray(cmp_)
        im.thumbnail((2400, 900))
        im.save(out / "preview" / "reference_check.png")
        Image.fromarray(tex).save(out / "preview" / "atlas.png")
    (out / "game.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "report.md").write_text(_report(manifest, opts), encoding="utf-8")
    log(f"[6/6] done in {manifest['seconds']} s -> {out}  (reference view: IoU {iou:.4f}, PSNR {min(psnr, 99):.1f} dB)")
    return manifest


# ------------------------------------------------------------------------------------------------ report
_TIER = {
    "M": "M - mesh-lock: a generated mesh locked to the picture; the reference view is exact, the sides and back follow the mesh's own shape and their colours are solved on its surface",
    "A": "A - analytic: a real solid whose every side is consistent with the picture",
    "A/B": "A/B - analytic body (walls, stands, floor) plus relief patches for what rises above it; the unseen half is a turned copy of the seen half",
    "B": "B - proxy / relief: real volume is added and the reference view is exact, the hidden sides are an educated guess",
}
_ENGINE_HINTS = {
    "generic": "Collision hulls are in `collision/` (GLB + OBJ). Rename them to your engine's convention or re-export with `--engine godot|unreal`.",
    "godot": "Drop the .glb into the project: the `-convcolonly` nodes become convex collision shapes (they are not drawn). Godot builds LODs itself on import; the files in `lods/` are for comparison.",
    "unreal": "Import the .glb (or convert to FBX): the `UCX_` nodes are read as convex collision. Import `lods/*_lods.glb` for the `_LODn` chain.",
    "unity": "Import the .glb with glTFast / UnityGLTF; add MeshColliders from `collision/` (convex = on). Use `lods/*_lods.glb` (`_LODn` names) to build a LOD Group.",
    "ursina": "Use the OBJ: `Entity(model='obj/NAME.obj', texture='obj/NAME_albedo.png', collider='mesh')`, or `collider='box'` and the hulls in `collision/` for custom colliders.",
}


def _report(m: dict, opts: AssetOptions) -> str:
    L = [f"# {m['name']} - game asset pack", "",
         f"Made by Aura White {m['aura_white']} from `{Path(str(m['source'])).name}` in {m['seconds']} s.", "",
         "## What it is", f"- Subject kind: **{m['kind']}** - {m['kind_reason']}",
         f"- Quality tier: **{_TIER.get(m['tier'], m['tier'])}**",
         f"- Size: {m['size_m'][0]:.2f} x {m['size_m'][1]:.2f} x {m['size_m'][2]:.2f} m (X x Y x Z), units metres, +Y up, +Z = the side the picture shows, pivot at the {m['pivot']['type']}.",
         f"- Reference view: IoU {m['fidelity_reference_view']['iou']:.4f}, PSNR {m['fidelity_reference_view']['psnr_db']} dB against the original picture "
         "(99 dB = identical pixels).", "", "## Files"]
    for k, v in m["files"].items():
        if isinstance(v, dict):
            L += [f"- textures: " + ", ".join(f"`{p}`" for p in v.values())]
        else:
            L.append(f"- {k}: `{v}`")
    L += ["", "## Level of detail", "", "| LOD | triangles | vertices |", "|---|---:|---:|"]
    L += [f"| {l['name']} | {l['triangles']:,} | {l['vertices']:,} |" for l in m["lods"]]
    L += ["", f"One atlas ({m['atlas']['width']}x{m['atlas']['height']}) serves every LOD. Texel density at the reference view (texels per metre): "
          + ", ".join(f"{k} {v}" for k, v in m["atlas"]["texels_per_metre_at_reference_view"].items()) + ".",
          "", "## Checks", "", "| check | result | detail |", "|---|---|---|"]
    for c in m["checks"]:
        L.append(f"| {c['check']} | {'pass' if c['ok'] else ('n/a' if c['ok'] is None else 'WARN')} | {c['detail']} |")
    L += ["", f"UV overlap factor {m['atlas']['uv_overlap_factor']} (1.0 = unique UVs; values above 1 mean the two halves of the object share texture "
          "on purpose - let the engine generate lightmap UVs if you bake lighting).", "",
          "## Engine notes", _ENGINE_HINTS.get(opts.engine, ""), "",
          "## What to know (honest limits)"]
    L += [f"- {n}" for n in m["notes"]]
    L += ["- Only the side the picture shows is guaranteed to match it. For a better back or side, make a picture of that view with any "
          "image tool and pass it with `--back-image`.", ""]
    return "\n".join(L)
