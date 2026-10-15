# Aura White 8 - real 3D from text, pictures or a whole character sheet

```
aura-white make "a bronze dragon statue"             # text    -> reference pictures -> real 3D
aura-white make knight.png                           # picture -> real 3D, back and sides included
aura-white make front.png back.png side.png          # several pictures (named by file, or back=rear.png left@80=photo.png)
aura-white refine aura_out/knight --add back=b.png   # not happy? a NEW version; v1 is kept
aura-white doctor3d --deep                           # what can this machine run? + a real end-to-end self-test
```

Install: `install.bat cu126 3d` (Windows) or `./install.sh cu126 3d` (Linux/macOS), or by hand
`pip install -r requirements-3d.txt && aura-white setup --hunyuan`.  Then `aura-white doctor3d --deep`.

## New in v8: your own sheet -> a model, and Aura's own model

```
aura-white sheet --front f.png --back b.png --left l.png --right r.png --height 1.68    # no GPU, no model download
aura-white sheet --mesh generator_output.glb --front f.png --back b.png --left l.png --right r.png   # repaint any mesh from the sheet
aura-white native licences | budget | make-data | teach | train-vae | train-flow | evolve | distill | bench | approve
```

`sheet` carves a closed, textured GLB from outlines of every side (back from the back picture, sides from the side pictures) and
reports how well it reproduces each picture.  Read the limits in `NATIVE.md` section 5: loose clothing and long hair give a blockout, and
the best finish is a generator's mesh painted from your sheet (`--mesh`).  `native` is Aura's own trainable model with a licence ledger,
quality gates, a frozen benchmark, teachers, self-improvement and compression - `NATIVE.md` says what was measured and what was not.

### Why the old output was a flat cut-out, and what changed

Three separate causes, each fixed:

1. **Slab geometry.**  `relief` / `source_locked` / the analytic figure builders take the front from the picture and mirror it for the
   back.  `studio`'s automatic mode even *chose* relief over the real model by comparing front-view IoU - a slab matches the front
   perfectly by construction, so it always won.  Every neural back-end needed 9-24 GB, so on an 8 GB card nothing qualified and
   the slab was silently used.  **Now:** `make` always builds a volume (Hunyuan3D-2 shape model, which runs in <= 8 GB, see below),
   `asset` generates a real mesh for characters / props / buildings, `studio` defaults to `standard`, and the auto mode no longer
   ranks a slab against a volume.  If no true-3D generator can run, it **stops and tells you what to install** - it never hands you a slab
   (`--allow-basic` accepts the built-in network or a visual hull, clearly labelled; `asset --allow-flat` restores the old relief).
2. **A wrong outline.**  When matting failed the old code used *the whole picture* as the object - a rectangle, extruded.  **Now**
   `gen3d/outline.py` runs every method that applies (alpha channel, exact plain-background matte, border flood, line-art interior
   fill, isnet-anime / birefnet / isnet-general / u2net via `rembg`, GrabCut), scores each *without a ground truth* (does the boundary sit on real
   edges, one solid piece, no leak, do independent methods agree), takes a majority vote, refines with a guided filter, and
   **refuses** (`OutlineError`) rather than guess.  A real alpha channel always wins outright.  Outline previews are written to `views/outline_*.png` - look at them first.
3. **No information about the back.**  Colour of unseen sides was a smooth fill.  **Now** pictures you give (back, sides) are registered to
   the mesh by silhouette and projected into the atlas; for the rest, a depth-conditioned painter (SDXL + depth ControlNet + IP-Adapter)
   paints the *mesh's own* back and sides from its depth, carrying the reference's colours.  The reference view itself stays pixel-exact.

### Same quality on every machine

Quality is a **recipe** (`--quality draft|standard|high|ultra`): steps, mesh resolution, how many candidates are compared, how closely
the mesh must agree with your pictures.  The hardware planner (`gen3d/hardware.py`) may only change *how* a stage runs - resident,
whole-model offload, layer streaming, or the processor, smaller chunks - never *what* runs.  Every job records the recipe's quality
signature, so you can check two machines used identical settings.  If a stage runs out of memory it is re-planned (more offload, smaller chunks)
and retried; it never silently drops to a weaker method.  `aura-white doctor3d` prints the plan for your card and for 4 / 6 / 8 / 12 / 24 GB.

| GPU memory | shape model (Hunyuan3D-2, 1.1 B) | what differs |
|---|---|---|
| 12 GB+ | resident | fastest |
| 8 GB | resident, or model offload if other programs hold memory | a little slower |
| 4-6 GB | model offload (peak ~2.6-3.6 GB estimated) | slower |
| no GPU | processor, 16 GB+ RAM | much slower, **same result** |

### How a result is judged (and re-rolled)

`gen3d/controller.py` generates several candidates per round and, for each: orients it (generators disagree about up/front: the 24 axis
rotations are tested against your pictures), fits a camera per picture by silhouette IoU, **fuses** it with every picture
(`fusion.py`: voxel carving + hull-limited growth, so a dropped tail comes back and a stray lump is removed, without touching a mesh that
already agrees), and **audits** it (`audit.py`: depth / width, thickness, mid-surface and mirror correlation, open edges, loose
fragments).  The score is *not* front IoU alone.  A flat result raises the guidance and re-seeds; a loose outline raises the mesh grid; swapped left/right
labels are detected and the generator is re-run with them exchanged.  The best candidate is kept and the report says plainly whether it passed.

### Every run is a folder; refining never overwrites

`aura_out/<name>/` holds `job.json` (request, recipe + signature, every version's metrics), `inputs/`, `views/` (outline previews, the
other text-to-image candidates), `v1/ v2/ ...` (`mesh.glb`, `report.json`, the asset pack) and `report.md`.  `refine` adds pictures (a new
picture **replaces** the stored one for that view), picks another text-to-image candidate (`--pick N`), changes seeds or quality - and
produces v2 next to v1.

### What has been verified (and what has not) - read this

Verified in the build environment (no GPU, no torch, no network): the voxel / fusion / audit / controller / outline / intake / planner /
texture-projection logic, the whole engine and asset pack end to end with a **fake shape generator that misbehaves on purpose** (flat
first candidate, rotated frame, swapped labels), the refusal paths, the CLI wiring, 58 new test functions (`tests/test_gen3d_*.py`) plus the
existing torch-free tests.  **Not** verified here: the real Hunyuan3D-2, SDXL, ControlNet and IP-Adapter weights running.  Those adapters
follow the vendors' documented APIs and filter arguments against the real signatures, but the only proof is on your machine:
`aura-white doctor3d --deep` loads the shape model and builds a mesh.  The 8 GB figures above are estimates from the vendor's ~6 GB
number plus offload arithmetic; the planner's out-of-memory retry exists because estimates can be wrong.  Hard real-world pictures (a
character on a busy photograph) are best served by the learned segmenters: install `rembg`, and check `views/outline_*.png`.
Nothing here promises a perfect model from every input - what it promises is that a weak result is *measured and reported*, and that
you have the levers (`--add`, `--pick`, `--quality ultra`) to improve it.

Licences matter: Hunyuan3D-2's community licence excludes some territories and has a service threshold - see `NOTICE.md`.  `--allow-basic`
keeps a fully MIT path (built-in network / visual hull) available at lower quality.

## Game assets (v6): `aura-white asset`  (sample previews and reports in `samples/`)

One picture in, a folder a game engine can use out - for a character, a prop, a building, a sphere / moon, an arena, a
floating island:

```
aura-white asset picture.png                           # analytic / relief solid, no model needed
aura-white asset picture.png --proxy mesh.glb          # lock a generated mesh to the picture (best sides and back)
aura-white asset picture.png --proxy mesh.glb --back-image back.png
aura-white asset picture.png --kind prop --height 1.8 --engine godot -o packs/scythe
```

The pack contains `NAME.glb` (LOD0, PBR maps, collision hulls for `--engine godot|unreal|unity|ursina`), `NAME_unlit.glb`,
`lods/NAME_lods.glb` (four LODs with `MSFT_lod`, one shared atlas), `obj/NAME.obj` + PNG (Ursina: `Entity(model='NAME',
texture='NAME_albedo.png')`), `textures/` (albedo, normal, occlusion-roughness-metal, emissive), `collision/`, `game.json`,
`report.md` and `preview/` sheets.  Units are metres, +Y up, +Z front, pivot at the base (`--pivot`).

**What is guaranteed.**  The reference view (the picture's own view) is re-rendered from the finished asset and compared with
the picture; IoU, outline error and PSNR are in the report.  **What is not.**  The sides and the back are never "reconstructed
from the picture" - that information does not exist.  They come from (in order of quality):

0. (always) the part of the object the picture shows head-on is textured with the picture itself at 1:1 (a "front chart"),
   so the reference view is reproduced texel for pixel; the rest of the surface is coloured as below.  Under `--proxy`,
   `--back auto` echoes the near side onto the far side for props, buildings, arenas and islands (characters stay `soft`);
   `--back soft|mirror|flat|echo` choose it explicitly.
1. a mesh from a generator on your GPU - Hunyuan3D-2.1, TRELLIS.2, Meshy, Tripo, Rodin ... - passed with `--proxy`; its own
   shape is used for the sides and back, its own texture colours them (`--no-proxy-colour` to ignore), and the picture is
   locked on top;
2. a picture of the object from behind (`--back-image`, any image generator, same framing);
3. the built-in solids (revolve / relief / arena / island / sphere), whose hidden half is an educated guess.

Workflow for the best result: (1) make a mesh from the picture with Hunyuan3D-2.1 or TRELLIS.2 on a GPU (or Meshy / Tripo in
the browser) and export GLB with texture; (2) `aura-white asset picture.png --proxy that.glb`; (3) read `report.md`.

Optional packages (`pip install ".[assets]"`): `xatlas` (real UV islands), `fast-simplification` (quadric LODs),
`trimesh` (PLY / STL proxies).  Without them the asset still builds (box-projection UVs, clustering LODs) and says so.

## Aura White Studio (v4): render, compare, correct - and joint multi-view

v4 closes the gap between "technically valid" and "actually matches the picture": it now measures its own result
against the source and either fixes what's wrong or tells you honestly that it couldn't, instead of shipping a
guess and calling it done.

```
run.bat studio art.png                                  # DEFAULT = source_locked (see below); auto-compares the exact 'relief' geometry against the
                                                         # built-in model when no GPU shape model is set up, keeps
                                                         # whichever matches the artwork better (was: always fell
                                                         # back to the weak built-in model - see below)
run.bat studio art.png --fidelity source_locked         # (the default) the picture IS the reference view: exact
                                                         # silhouette (IoU ~0.9999), exact pixels, graded publication
run.bat studio art.png --fidelity standard              # the older generic pipeline (generated side views, atlas)
run.bat studio art.png --clean-rim                      # source_locked: swap the pale 1-px old-background blend on the
                                                         # outline for object colour (nicer on dark viewers)
run.bat studio front.png --view back=back.png --view right@90=right.png   # joint multi-view reconstruction
run.bat studio art.png --lods 3 --collision             # LOD0-3 + a convex-hull collision mesh alongside the asset
run.bat studio art.png --blender-bake                   # real ambient occlusion baked with a local Blender install
run.bat lod existing_asset.glb --levels 3 --collision convex
run.bat repair existing_asset.glb                       # standalone topology fixer for any mesh
```

**What was actually wrong, and what changed:**

- **Auto mode was quietly using the weak model.** `--geometry auto` tried Pixal3D -> TRELLIS.2 -> Hunyuan3D-2.1 ->
  InstantMesh -> the built-in model, in that order - but never tried the GPU-free, silhouette-exact `relief`
  backend before falling all the way back to the built-in one. On a flat-art test case this alone was the
  difference between a 0.80 and a 0.92+ silhouette match. Auto mode now compares `relief` against the built-in
  model (by actual measured silhouette agreement, not a guess) whenever no GPU shape model is available, and keeps
  the winner - this is the same mechanism `--best-of` already offered, now applied automatically where it matters.
- **Render -> compare -> correct.** After texturing, the pipeline renders the finished, textured mesh from the
  artwork's own recovered camera and measures it against the artwork directly (silhouette IoU + PSNR) -
  `asset_report.json["source_fidelity_iou"]` / `["source_fidelity_psnr_db"]`, always, not just in source-locked
  mode. If the outline doesn't yet meet `--refine-target-iou` (default 0.985), it spends up to `--refine-iterations`
  extra passes re-snapping at higher resolution before finalizing (`asset_report.json["refine"]
  ["render_compare_correct"]` shows each pass). `asset_difference.png` is a new preview showing exactly where the
  final render still disagrees with the source.
- **The fidelity gate is graded, not all-or-nothing.** `--fidelity source_locked` still tries to reach
  `--fidelity-target-iou` (default 0.999; the builder adds vertices wherever mask pixels are still uncovered), but a result that falls short is shipped
  with `source_fidelity_target_met: false` and a clear note (and an honest `asset_difference.png`) instead of
  refusing outright. Pass `--strict-fidelity` to get the old all-or-nothing behaviour back.
- **The source-locked shell is actually closed now.** Its non-manifold side-wall construction (an edge-winding bug
  that a bundled test caught but a quoted SHA-256 hash didn't) is fixed by routing the shell through the same
  topology repair used everywhere else, plus two repair engine bugs this surfaced: `boundary_loops` could silently
  drop a loop at a "pinch" vertex where two loops touch at one point, and ear-clipping a self-touching loop could
  net *more* open edges than it closed on dense meshes - both fixed (`mesh/repair.py`: edge-visited loop tracing,
  pinch-splitting before ear-clipping, and a best-state fallback if a pass ever makes things worse instead of
  better). `aura-white repair` exposes this fixer for any mesh, not just Aura White's own output.
- **Real PBR, not a placeholder.** GLB export now writes an ORM texture (R=occlusion, G=roughness, B=metallic,
  the glTF convention) and, for `--fidelity source_locked`, `KHR_materials_unlit` so a viewer's lighting can't
  recolour the locked-in source pixels. `--blender-bake` replaces the occlusion channel's flat 255 with a real
  Cycles ambient-occlusion bake from a local Blender install (`AURA_WHITE_BLENDER` env var, or PATH/common install
  dirs) - Blender ships QuadriFlow quad-retopology and an OpenVDB voxel remesher too
  (`aura_white.backends.blender_tools.quad_remesh` / `voxel_remesh`), reached the same way Pixal3D/TRELLIS.2 are:
  as a subprocess in Blender's *own* Python, so a Blender version can never conflict with anything else here.
- **LODs and a collision mesh** (`--lods N`, `--collision`, or standalone via `aura-white lod`): each LOD is
  decimated straight from the original mesh (never chained from a coarser LOD, so error doesn't compound) and
  independently topology-checked; the collision mesh is a closed convex hull by default, or a closer (non-convex)
  Blender voxel-remesh fit when asked for one.
- **Joint multi-view reconstruction** (`--view name=path`, repeatable; or `AuraStudio.generate_multiview`): give it
  a front photo plus a back/side photo (or more) of the *same* object and it registers every view against one
  shared mesh, pulls the silhouette to agree with **all** of them at once (`forge/snap.py:
  snap_to_silhouettes`), and bakes the texture from every picture instead of guessing the unseen side - on a test
  object this measurably beats front-photo-only on a back-view render (see `tests/test_multiview_studio.py`).
  Azimuths for `front`/`back`/`left`/`right`/`3-4-front`/`3-4-back` are guessed automatically; anything else needs
  `--view name@degrees=path`. This does not solve for a different camera height per photo - shots from noticeably
  different elevations will register worse than a matched turntable rig.

None of this changes what v3 already got right (matte, chain/cord detection, residual completion, topology repair
basics) - v4 adds the measurement loop and the multi-view path on top of it.


The original pipeline colours *vertices*, so fine detail is smeared into "melted clay". **`aura-white studio`** builds
what commercial services do - a UV-unwrapped mesh with a 2K-4K albedo texture and a normal map - from open models,
puts the **original artwork's own pixels** on every surface it can see, then **repairs, snaps and completes** the
result so hairline chains, cords, ribbons and gems that every neural model drops are no longer missing:

```
artwork ─┬─> matte       exact geometric matte (flat backgrounds incl. white-on-white) | rembg | transparent PNG as-is
         ├─> geometry    Pixal3D | TRELLIS.2 | Hunyuan3D-2.1 | InstantMesh | relief (GPU-free) | built-in Aura White
         │               (auto: the best that can run here; --best-of runs several and keeps the one whose
         │                silhouette matches the artwork best)
         ├─> side views  Zero123++ (only for the far side of the object)
         └─> registration: the artwork's camera is recovered by silhouette matching + stereo consistency
   mesh ── topology repair ── silhouette snap ── residual completion (chains/cords/ribbons/gems) ──
        decimate ── UV unwrap ── multi-view bake (+ mirrored-artwork back) ── normal map ── GLB / OBJ+MTL
```

```
run.bat studio art.png                              # auto geometry, auto matting, repair+snap+complete all on
run.bat studio art.png --best-of pixal3d trellis2 hunyuan3d --quality high
run.bat studio art.png --geometry relief             # GPU-free, silhouette-exact - best for flat/thin subjects
                                                      # (weapons, wings, emblems, cut-out characters); real
                                                      # interlocking chain links and round cords, not blobs
run.bat studio art.png --matting exact               # force the geometric matte (keeps hairlines, no ML model)
run.bat studio art.png --no-complete --no-snap        # geometry + texture only, no post-processing
run.bat setup --repos pixal3d trellis2 hunyuan3d      # clone the upstream repos into third_party/
```

**Why chains stopped disappearing.** Every feed-forward image-to-3D network (TripoSR, TRELLIS, Hunyuan3D, ...) drops a
2-pixel-wide chain: a marching-cubes/FlexiCubes threshold cannot hold a feature that thin, and the mesh-cleanup pass
that keeps output watertight throws small disconnected parts away as noise. Instead of trying to force a diffusion
model to hallucinate correct topology from one picture, Aura White finds what the mesh is missing directly in the
**artwork's own camera** (`aura_white/forge/complete.py`): pixels the artwork shows that no mesh surface projects onto.
A skeleton-based detector (`forge/wires.py`) tells a hairline chain from a ribbon or a fin by how its inscribed radius
varies along its length, and builds **real interlocking stadium links** (or a round tapered cord for laces/vines) at
the correct depth and width; everything else missing (gems, studs, small fins) becomes a silhouette-exact relief patch
(`forge/patch.py`, marching-squares + Delaunay, holes included). `--geometry relief` uses the same engine as the whole
shape, for subjects (weapons, banners, emblems, 2D characters used as props) where an exact 2.5-D reconstruction beats
an approximate 3-D guess - no GPU, no model download, runs in a couple of seconds.

**Registration is then used twice more.** `forge/snap.py` pulls the mesh's own silhouette onto the artwork's silhouette
(within the image plane only, relaxed smoothly over the surface) before texturing, so a blade tip or a fingertip a
neural model rounded off lines up exactly with the picture. `mesh/repair.py` is a from-scratch, fully vectorised
topology fixer (no trimesh/pymeshlab/open3d): non-manifold edges are cut by keeping the most-coplanar pair of faces
and splitting the rest, "bow-tie" vertices are split by connected-components over face corners, boundary loops are
closed with ear-clipping, orientation is made consistent per shell by BFS + signed volume, and dust islands are
dropped - all reported in `*_report.json["refine"]` so a bad repair is visible, never silent.

Why it stays crisp: the bake weights the original picture 8x over generated views, so wherever the artwork can see the
surface the texture *is* the artwork (full resolution, exact colours); Zero123++ views (only 320 px) fill in what the
artwork cannot see, colour-matched to it; for the `relief` backend the far side is instead textured with the artwork
**mirrored** across the object's own depth axis (`--back mirror`, on by default for `relief`) rather than blurred
nearest-neighbour fill, which is the correct answer for the many game-art weapons/props that are left-right symmetric
through their thickness. Texels nothing can see are completed from their nearest coloured neighbours in 3D (seamless
across UV islands). Silhouette/occlusion-aware weights stop background colour bleeding at the edges.

**Never breaks:** the texturing engine (`aura_white/raster`, `aura_white/texture`) is pure PyTorch/NumPy, and the new
`forge`/`mesh.repair`/`evaluation` modules are pure NumPy/SciPy - no compiled extension, no native wheel, nothing that
can segfault the interpreter. `xatlas` and `fast-simplification` still run **in a child process** and fall back to the
built-in equivalents on any misbehaviour. Pixal3D and TRELLIS.2 run **as a subprocess in their own Python
environment** (`AURA_WHITE_PIXAL3D_PYTHON` / `AURA_WHITE_TRELLIS2_PYTHON`), so their pinned CUDA/torch versions can
never conflict with Aura White's or with each other's; a crash there is caught, reported, and the pipeline falls back
to the next geometry backend. Every heavy model is optional, lazily imported, loaded, used and freed in turn.

Colab (free T4 works with `--quality standard`): open `notebooks/aura_white_studio_colab.ipynb`.
Extras: `pip install -r requirements-studio.txt` (or `install.bat cu126 studio`). Sources of the model repositories go in
`third_party/` (see `third_party/README.md`); Pixal3D and TRELLIS.2 keep their own virtualenv (`setup.sh` in each repo)
since they pin exact CUDA/torch versions - point `AURA_WHITE_PIXAL3D_DIR`/`_PYTHON` (and `_TRELLIS2_`) at them once
built. **Licences:** Zero123++ weights are non-commercial (CC-BY-NC) - use `--multiview off` for commercial work;
Hunyuan3D-2 is usable commercially outside the EU/UK/South Korea; TRELLIS.2 and TripoSR are MIT. See `NOTICE.md`.

Python:

```python
from aura_white.studio import AuraStudio, StudioOptions
studio = AuraStudio(options=StudioOptions(quality="high", best_of=("pixal3d", "trellis2", "hunyuan3d")))
result = studio.generate("art.png", progress=lambda stage, frac, msg: print(stage, msg))
result.save_all("out", "hero", formats=("glb", "obj"))         # + albedo/normal PNGs, previews, report.json
print(result.report["reference_iou"], result.report["notes"])  # how well the artwork matched the mesh, fallbacks taken
print(result.report["refine"])                                 # repair / snap / completion stats
```

The `*_report.json` and `*_registration.png` (artwork vs mesh silhouette: red = mesh sticks out, blue = artwork not covered)
tell you at a glance whether a result can be trusted.

### Other new commands

```
run.bat forge art.png -o out.glb           # relief backend alone: picture -> exact mesh in ~1-3s, no GPU
run.bat repair mesh.glb                    # fix non-manifold edges/vertices, holes, dust in any mesh
run.bat eval --pred outputs --gt ground_truth   # chamfer / F-score / normal consistency / voxel IoU, batch report
run.bat shape-train --steps 20000 --data procedural   # train Aura Shape (see below) from scratch, no dataset needed
run.bat shape-train --data /path/to/glb_or_obj_folder  # ...or on your own mesh collection (Objaverse, etc.)
run.bat shape-complete mesh.glb --ckpt aura_shape.pt   # fill in the hidden volume of a single-view mesh
```

### Aura Shape: an owned model for volume no picture shows

`aura_white/shape/` is a small conditional **flow-matching Diffusion Transformer over a truncated-SDF volume**
(`ShapeDiT` + rectified-flow training/sampling, `model.py`), trained from scratch - no borrowed weights. Given the part
of an object's volume a single view can observe, it generates the rest, with the observed voxels held exactly
consistent throughout sampling (`FlowMatching.sample`). It trains on an unlimited built-in procedural-shape generator
(random unions of primitives, `shape/sdf.py`) so `shape-train --data procedural` needs no dataset at all, or on any
folder of `.glb`/`.obj` meshes (Objaverse, your own asset library, ...) via `--data <folder>`. This is a genuinely
different tool from the residual completion above: completion rebuilds parts *visible in the picture* that the mesh
lost; Aura Shape guesses volume *no picture shows at all*. Both are optional and off by default in the main pipeline;
use `shape-complete` directly, or call `aura_white.shape.ShapeCompleter` from Python.

## What broke in TripoSR, and what replaced it

| TripoSR (original)                                   | Problem on a current Python stack                                        | Aura White |
|------------------------------------------------------|--------------------------------------------------------------------------|------------|
| `torchmcubes` (git, compiled)                        | Abandoned; needs a compiler; fails to build on new Python                | Built-in **surface-nets mesher** (numpy, ~100 lines) |
| `transformers==4.35.0`                               | Newer releases renamed the ViT layers, so the official checkpoint no longer loads | Own DINO ViT in plain PyTorch + a key normaliser (accepts old *and* new names) |
| `Pillow==10.1.0`, `omegaconf`, `einops`, `trimesh==4.0.5`, `xatlas==0.0.9`, `moderngl==5.10.0` | Exact pins / compiled wheels that don't exist for new Pythons | Not needed (own exporters, no UV baking) |
| `gradio`                                             | API changes every release                                                | Standard-library web server + WebGL viewer |
| `rembg` required                                     | onnxruntime/numba lag behind new Pythons                                 | Optional; built-in matte for plain backgrounds; transparent PNGs used as-is |
| `torch.cross(a, b)` without `dim`                    | Picks the wrong axis when there happen to be exactly 3 views             | Explicit `dim=-1` (`torch.linalg.cross`) |

Every version-sensitive spot lives in `aura_white/_compat.py` behind feature detection.

## Better than the original

* **No compiled parts.** Nothing to build, nothing that can fail to build.
* **Watertight, manifold, outward-facing meshes** (surface nets + checkerboard-face repair), with fewer, better-shaped triangles than marching cubes. Mesh is closed even if the object touches the bounding box.
* **5-10x fewer network queries**: a coarse pass finds the surface shell and only that shell is evaluated at full resolution (`--dense` gives the reference behaviour).
* **Floating specks removed**, light volume-preserving smoothing, degenerate faces dropped.
* **Half-size, pickle-free weights**: one-time conversion to fp16 `.safetensors` (about 0.84 GB instead of 1.68 GB). Re-uses a TripoSR checkpoint already on your PC (Hugging Face cache) - no second download.
* **Deterministic**, thread-safe API, `--low-memory` mode, CPU / CUDA / Apple MPS, fp32 (default) / fp16 / bf16.
* **`aura-white doctor`**: checks your install and runs a full self-test in seconds, so a Python/PyTorch update that breaks something is caught immediately.

## Install

**Windows:** double-click `install.bat` (or run it in a terminal).
NVIDIA GPU: `install.bat cu126` (use the tag shown on https://pytorch.org/get-started/locally/ for your CUDA version).
**Linux / macOS:** `./install.sh` (or `./install.sh cu126`).

Manual: `pip install -e ".[recommended]"` inside any virtual environment with Python 3.9 or newer.

Then check it: `run.bat doctor` (Windows) / `.venv/bin/aura-white doctor`.

## Use

```
run.bat photo.png                       # -> aura_out/photo/mesh.glb + mesh.obj
run.bat photo.png other.jpg --format glb obj ply stl --resolution 320
run.bat serve                           # local web app at http://127.0.0.1:7860 with a 3D viewer
```

The first real run prepares the weights automatically (or run `run.bat fetch`). If you ever used
TripoSR before, its `model.ckpt` is found and reused; otherwise about 1.7 GB is downloaded once.
Use `--weights PATH` to point at any `model.ckpt` / `.safetensors` yourself.

Python:

```python
from aura_white import AuraWhite

engine = AuraWhite.from_pretrained()                     # device="auto", precision="auto" (= fp32)
result = engine.generate("photo.png", resolution=256)    # path, bytes, PIL image or array
result.save("out.glb")                                   # or .obj / .ply / .stl
print(result.stats["vertices"], result.timings["total"])
```

### Options (`generate`)

| option | default | meaning |
|---|---|---|
| `--resolution` | 256 | grid size (64-512). Higher = more detail, slower |
| `--threshold` | 25 | density level of the surface (TripoSR default). Lower = fatter, higher = thinner; "No surface found" -> lower it |
| `--smooth` | 2 | Taubin smoothing passes (0 = off) |
| `--min-part-ratio` | 0.02 | drop floating pieces smaller than this share of the main body (0 keeps everything) |
| `--no-remove-bg` | off | image is already cut out / clean |
| `--matting` | auto | `auto` (alpha -> rembg -> built-in) / `rembg` / `border` |
| `--foreground-ratio` | 0.85 | how much of the frame the object fills |
| `--dense` | off | evaluate the whole grid (reference, slower) |
| `--up`, `--yaw` | y, 0 | output axis convention and extra rotation about it |
| `--device`, `--precision` | auto, auto | `cpu`/`cuda`/`mps`; `fp32`/`fp16`/`bf16` (fp16 = lower VRAM, CUDA) |
| `--turntable` | off | also render a turntable GIF (slow on CPU) |

Photos work best with one object, plain background, three-quarter view. Transparent PNGs skip background removal entirely.

## What has been verified (and what has not)

Verified by the test suite (`python -m pytest`, 71 tests) and `tools/parity_check.py`:

* **Architecture equals TripoSR's**: at full size both networks have 547 tensors / 418.7 M parameters with identical names and shapes (built from the published `config.yaml` values), so the official checkpoint loads without renaming.
* **Numerically equal to the original code**: with identical random weights, scene codes, densities, colours and rendered views match the original TripoSR source to float precision (tiny-size model; includes the position-embedding resize path). Perturbing any stage is detected, so the comparison is not vacuous.
* **Mesher**: sphere/torus/noise-field tests for closed, consistently oriented, edge-manifold output; adaptive evaluation reproduces dense evaluation (identical inside/outside classification on analytic scenes) while using about 9% of the queries at 256.
* **Pipeline end to end** with a random tiny network: preprocessing, encoding, meshing, colouring, all four export formats (read back with `trimesh`), CLI, web server job flow, weights conversion / loading (pickle, safetensors, fp16), resumable download.
* **Python versions**: the numpy-side (mesher, volume, image, exporters; 47 tests) passes on Python 3.9, 3.12, 3.13 and 3.14; the PyTorch side was run on Python 3.12 with PyTorch 2.14 / NumPy 2.5 / Pillow 12.

Not verified in the build environment (no access to the weights or a GPU there):

* Output quality with the **real pretrained weights** - run `python tools/parity_check.py --triposr <TripoSR-main folder> --real <folder with config.yaml + model.ckpt> --image photo.png` to compare against the original code, or just compare a mesh from both.
* The default `--up y` orientation puts the side the photo was taken from toward +Z; if a model comes out facing away, use `--yaw 180`.
* The web viewer's WebGL rendering in a real browser (its data format and script were checked, but not drawn).
* PyTorch on Python 3.13/3.14, CUDA, and Apple MPS.

## Layout

```
aura_white/
  model/      dino.py transformer.py triplane.py renderer.py net.py   (pure PyTorch, TripoSR-compatible)
  mesh/       surface_nets.py cleanup.py export.py repair.py lod.py   (numpy/scipy only; repair.py: topology fixer,
                                                                        lod.py: LOD chain + collision mesh)
  shape/      sdf.py model.py complete.py train.py                    (Aura Shape: owned flow-matching DiT for hidden volume)
  forge/      morph.py patch.py wires.py complete.py snap.py softraster.py relief.py   (image-space exact geometry;
                                                                        numpy/scipy only; snap.py: single- and
                                                                        multi-view silhouette snapping)
  image/      preprocess.py matte.py                                  (Pillow + numpy; matte.py: exact geometric matting)
  fidelity.py source-locked reference-view mesh (--fidelity source_locked)
  volume.py   coarse-to-fine density evaluation
  pipeline.py AuraWhite / MeshResult          weights.py  find, download, convert
  server.py viewer.py  web app                cli.py      studio | generate | serve | fetch | convert | setup | doctor |
                                                          forge | repair | eval | lod | shape-train | shape-complete
  raster/     torch_raster.py (exact z-buffer, no extensions) nvdiff.py (optional) camera.py
  texture/    unwrap decimate register bake fill normals preview  (UV atlas, camera fitting, multi-view baking)
  backends/   geometry.py (selection/auto/best-of) external.py (Pixal3D, TRELLIS.2: own-env subprocess) glb.py
              (dependency-free glTF reader/writer, now with UV) blender_tools.py (AO bake, quad/voxel remesh via a
              local Blender, own-process) hunyuan3d.py instantmesh.py multiview.py (Zero123++) setup_repos.py
  evaluation.py   chamfer / F-score / normal consistency / voxel IoU / silhouette IoU / PSNR, batch driver
  studio.py   AuraStudio: generate() (single image) + generate_multiview() (joint reconstruction)
              _isolate.py  run fragile native code in a child process
tools/parity_check.py   comparison against the original TripoSR source
tests/                  pytest suite
```

## Troubleshooting

* `torch NOT AVAILABLE` in `doctor`: install PyTorch for your Python from https://pytorch.org/get-started/locally/.
* "No surface found": lower `--threshold` (try 10), or use a clearer photo.
* Rough cut-out on a busy background: use a transparent PNG, or `pip install rembg`.
* Out of memory: add `--low-memory` (Python: `low_memory=True`) or lower `--resolution`.
* Set `AURA_WHITE_DEBUG=1` to see full tracebacks; `AURA_WHITE_HOME` moves the weights cache.

## Credits and licence

MIT licence (see `LICENSE`). Network design and weights: TripoSR by Tripo AI and Stability AI (MIT); see `NOTICE.md`.


## Source-locked mode (v5): what it guarantees and what it assumes

* **Reference view = the picture.** The front surface is a triangulation of the exact foreground mask (boundary vertex on
  every pixel edge of the 0.5 iso-line, so islands, holes and 1-px chains survive), UVs are the picture's own pixel
  coordinates, and the texture keeps the picture's native size.  Measured on a thin, detailed scythe: silhouette IoU
  0.9999, colour PSNR 52 dB (the rest is the soft edge band the mesh intentionally does not cover).
* **Back and sides are assumed**, not reconstructed: the back is the mirrored artwork, side walls sample colour a few
  pixels inside the outline.  They look like a thick cut-out from the side - that is the contract.  See `IMPROVEMENT_PLAN.md`
  for the path to real side/back geometry.
* The exported mesh has texture seams along the rim (duplicated vertices with identical positions); weld by position for
  physics/printing.  The few places where thin parts of the artwork touch are intentional pinch points.
* Developer extras: `pip install -r requirements-dev.txt` (pytest, trimesh).
