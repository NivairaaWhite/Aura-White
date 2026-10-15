# Changelog

## v8 - Aura's own model, and a no-GPU sheet reconstruction
- **New `aura_white.native`** (see `NATIVE.md`): Aura's own shape VAE and rectified-flow generator, a licence ledger (Hunyuan3D is
  refused as a teacher; its licence forbids it), procedural ground-truth factory, ground-truth-free quality gates (`gates.py`), a
  frozen benchmark that separates oracle from verifier metrics, licensed teachers (`teach.py`), verifier-gated self-improvement
  (`evolve.py`), compression with measured effects (`compress.py`: budget formula, depth pruning + distillation, guidance
  distillation, int8), checkpoints as plain tensors (`weights_only`).  CLI: `aura-white native ...`.
- **New `aura-white sheet`** (`native/multiview.py`): front/back/left/right pictures -> textured GLB with a real back and sides, no
  network and no GPU; or `--mesh` to keep a mesh from any generator and repaint it from the pictures (finds its orientation first).
  Rounded cross-sections (not boxes), thin-part depth limit, xatlas unwrap, occlusion-aware multi-view painting, outline agreement
  reported per view.  Honest limit documented: loose clothing / long hair give a blockout, not a finished character.
- `gen3d.shapegen.AuraNativeShape`: opt-in back-end; a checkpoint is used only after `aura-white native approve` (>= 5,000 real
  assets, passing benchmark, clean data) or with `AURA_NATIVE_ALLOW_UNAPPROVED=1`.
- Measured (CPU, tiny preset): self-improvement round 1 showed no gain on ground truth for any arm; guidance distillation gave
  about -30 % time at equal quality; int8 gave no gain at this size; 4 -> 3 block pruning lost quality without longer healing.
- Hardening found in review: the licence ledger resolves name variants (`Hunyuan3D_2.5`, `hunyuan3d-2mv`, ...) to the denied entry, so
  `accept_risk` can no longer admit them; `CommandTeacher` refuses a command that mentions Hunyuan; `sheet` reports an empty picture
  in plain words.
- Tests: `tests/test_native.py` (22 tests).  Extras: `pip install aura-white[sheet]`, `[native]`.

## v7 - real 3D (back and sides), any outline, same quality on every machine
- **New `aura-white make`**: text, one picture or several pictures -> a closed, textured 3D asset.  `refine` makes a new version
  (add pictures, pick another text-to-image candidate, other seeds, higher quality); `doctor3d [--deep]` shows the memory plan, which
  generators are ready, and self-tests the engine.  A bare picture path now runs `make`.
- **`aura_white.gen3d`** (torch is imported only by the stages that need it): `recipe` (quality chosen by you, signed), `hardware`
  (probe + per-stage memory planner + out-of-memory retry), `views` (one documented camera convention), `meshutil` (winding-number voxelisation,
  signed-distance remeshing), `fusion` (multi-view silhouette carving + hull-limited growth), `audit` (flat / mirrored-relief detection, topology),
  `controller` (candidates, frame calibration over the 24 axis rotations, left/right swap detection, re-roll policy), `outline`
  (multi-method, scored, never guesses), `intake` (labels from names), `textgen` (prompt + ranked reference candidates), `shapegen`
  (Hunyuan3D-2 / 2mv in <= 8 GB, built-in network and visual hull as labelled fall-backs, existing heavy back-ends under `--tier max`),
  `viewsynth` (depth-conditioned painter for unseen sides), `paint` (registers any view by silhouette and projects it into the atlas),
  `job` (non-destructive versions), `engine`.
- **Fixed - the flat cut-out**: `asset` generates a real mesh for characters / props / buildings (`generate="auto"` in the CLI;
  `--allow-flat` restores relief); `studio` CLI defaults to `--fidelity standard`; `studio` auto mode no longer ranks a relief slab against
  a volume by front IoU; a Hunyuan3D-2 back-end is registered for `studio`.  The library default `StudioOptions().fidelity` is unchanged.
- **Fixed - whole picture as the object**: new outline extractor (`matting="outline"`, also used by `asset` before its border/whole-picture
  fall-backs).  A real alpha channel wins outright; a picture with no trustworthy outline is refused.
- **Fixed**: `aura_white.texture` is lazy (numpy-only helpers no longer import torch, so the `asset` pack path runs without it);
  `asset` accepts `extra_views`; mesh-lock log line no longer prints the proxy array.
- Tests: `tests/test_gen3d_core.py`, `test_gen3d_quality.py`, `test_gen3d_engine.py`, shared `gen3d_fakes.py`, numpy-only `prims.py`.

## v6 - game assets, real sides and back
- New `aura-white asset PICTURE` command and `aura_white.asset` package: picture -> **game-ready pack** (`NAME.glb` with
  PBR albedo / normal / occlusion-roughness-metal / emissive maps, an unlit GLB, OBJ + PNG for Ursina and friends, four
  LODs with `MSFT_lod` sharing one atlas, convex collision hulls named for Godot / Unreal / generic engines,
  `game.json`, `report.md`, preview sheets).  Metres, +Y up, +Z front, pivot at the base.
- **Reference view = the picture, now measured**: an orthographic reference camera and an exact z-buffer render the asset
  back from the picture's own view; IoU, outline error (within 2 px) and PSNR are written into the report and checked.
- **Mesh-lock mode** (`--proxy mesh.glb`): take a mesh from any generator (Hunyuan3D-2.1, TRELLIS.2, TripoSR, Meshy,
  Tripo, Rodin ...), fit its pose to the picture's silhouette (yaw / elevation / scale by soft-IoU optimisation), snap
  its outline into the matte, grow it out to the matte edge, add thin relief patches for picture pixels it does not
  cover, unwrap with xatlas, and bake colour so that the picture colours everything the camera sees.  Sides and back keep
  the mesh's own shape; their colour is the mesh's own texture when it has one (`--no-proxy-colour` to ignore it), else a
  harmonic (Laplace) diffusion of the visible colours over the surface - no mirrored picture, no stretched streaks.
  A picture of the object from behind (`--back-image`) is used as extra colour data.
- **Picture front chart**: every face the reference camera sees head-on is textured by planar projection of the picture
  itself (UV = picture pixel), packed at 1:1 whenever the atlas has room, so the reference view is reproduced texel for
  pixel (fox 31 -> 37 dB, scythe 22 -> 28 dB, 33 dB away from the outline; arena / island default to a 4096 atlas for the same
  reason).  The rest of the surface is unwrapped with xatlas and coloured as described above.
- **`--back auto|soft|echo|mirror|flat`**: under mesh-lock props, buildings, arenas and islands now default to `echo` - the
  near side is reflected through the object's mid-plane and worn by the far side (a far wall carries the near wall's arches),
  blended smoothly over the flanks; characters keep `soft` (an echo would put a face on the back of the head).
- High-poly to low-poly **normal and cavity-AO bake** (tangent space, KD-tree) so a 40k-triangle LOD0 keeps the detail of a
  500k-triangle source.  Quadric decimation (`fast-simplification`) with a pure NumPy clustering fallback.
- Analytic builders (no mesh needed): sphere / moon (limb band rebuilt by reflection - no streaks at the side views),
  figure and prop relief solids, arena (ring wall + stands + floor), floating island.
- Robustness: only numpy + Pillow + torch required, scipy core; xatlas / fast-simplification / trimesh optional and
  guarded (box-projection UVs and vertex-clustering LODs are the fallbacks).
- `--kind sphere` on a picture that already has an alpha now works (outline ellipse fitted from the matte).
- Tests: `tests/test_asset.py` (camera round trip, rasteriser, mesh io, glTF writer, harmonic colours, pose fit, outline
  lock, convex hulls, mesh-lock and sphere end to end through the CLI).
- Versions: package 6.0.0 (was 4.0.0 in the code and 2.0.0 in pyproject); `pip install .[assets]` pulls the optional extras.

## v5 - exact-look
- Source-locked mode rebuilt: pixel-edge boundary vertices, native resolution, coverage loop, pinch splitting,
  directed-edge shell (scythe silhouette IoU 0.955 -> 0.99995, closed, no repair pass), wall UVs inset into the artwork.
- Pinch points (thin parts that touch) are nudged 0.01 px apart per vertex copy: exported mesh is watertight and
  manifold even after welding by position (trimesh: watertight, consistent winding).
- Texture: native size; every pixel the mesh can show is byte-identical to the input (colour PSNR 14.8 -> 52.7 dB on
  the scythe); `--clean-rim` opts in to replacing the pale anti-aliased rim.
- `source_locked` is the default for `studio` and `aura-white studio` (`--fidelity standard` for the old pipeline);
  scipy is a base dependency (falls back to `standard` with a note if missing).
- QA: final IoU measured at native resolution; retry bug (depth taken from the locked shell) removed with the coarse preflight.
- Removed dead code and duplicates (incl. duplicate repo discovery in `backends/_compat.py`); `aura_white.raster`
  imports torch lazily; `requirements-dev.txt` added.
- Tests: hard-mask regression (chain, wire, island, hole, pinch), wall-UV and clean-rim tests, a source-locked
  end-to-end test; generic-pipeline tests pinned to `fidelity="standard"`.
