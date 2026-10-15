> **v7 status.**  Done: true-3D default path (`make`), <= 8 GB execution of the full-quality recipe, multi-view fusion, flat detection,
> any-picture outlines, non-destructive refinement.  Open: the web app (`serve`) does not yet expose `make` / `refine` - use the CLI;
> the depth-conditioned painter is the least-tested part (no real weights in the build environment); a learned multi-view
> generator (e.g. an SDXL multi-view adapter) could replace per-view painting for tighter cross-view consistency.

# Aura White - plan v6: real sides and back, game-ready assets

Brief (v6): "the source lock is good for the front, but the sides and the back are not a good way".  Make the sides and
the back look like part of the same object, make the result usable in a game, and make it work for all five test pictures
(fox girl, scythe, marble arena, floating castle island, blood moon).

## What was wrong in v5 (audit)
| # | Finding | Effect |
|---|---------|--------|
| 1 | The relief / "source-locked" body is a **slab**: front = picture, sides = extruded rim, back = mirrored picture | side views are planks, the back is the front flipped |
| 2 | Single triangulated sheet of 110k-400k triangles, no LODs, no collision, no atlas rules, no units | not usable in an engine without hours of clean-up |
| 3 | No notion of *how well the reference view matches*: only an internal IoU on a half-resolution render | a bad fit could pass |
| 4 | Matte depended on a white or plain background | busy backgrounds (arena, island, moon with sparkles) failed |
| 5 | Fitting used a perspective camera for artwork that is drawn orthographically | geometry never lined up with the picture |
| 6 | Round things (moon) stretched the last 2-3 picture pixels over the whole limb band | vertical streaks when the ball turned |
| 7 | The neural hybrid (Hunyuan3D / TRELLIS shape + picture colour) was never run here (no GPU, weights unreachable) | untested claim |

## What v6 does about it
1. **Orthographic reference camera + exact z-buffer** (`asset/camera.py`): every layer is slid along view rays, so a pixel of
   the picture stays on the same surface point; the report renders the asset back from the picture's view and measures IoU,
   outline error and PSNR (colour error is also reported away from the outline).
2. **Mesh-lock** (`asset/meshlock.py`, `meshfill.py`, `hibake.py`): sides and back come from a real 3D shape produced by a
   generator, and the picture is locked onto it.  Pose fit by soft-IoU Nelder-Mead over yaw families; outline snap;
   grow-to-matte; residual relief cards; xatlas UVs; shared-atlas LODs; harmonic colour on the surface; the mesh's own
   texture for the unseen sides when it has one; high-to-low normal + cavity bake.
   Texturing is split in two: faces the reference camera sees head-on take the **picture itself** as their texture (planar
   projection, a chart packed at 1:1 - the reference view is exact there), everything else is unwrapped and coloured by
   diffusion (+ the mesh's own texture, + `echo` of the near side for props / buildings / arenas / islands, + `--back-image`).
3. **Game conventions** (`asset/pipeline.py`, `gltf.py`, `collision.py`, `validate.py`): metres, +Y up, base pivot, power-of-two
   atlas, UVs in 0..1, MSFT_lod, convex hulls named for the engine, unlit GLB, OBJ + PNG, `game.json` and checks.
4. **Analytic solids** where no mesh exists: sphere, arena, island, figure / prop reliefs, with fold / rotate charts.

## Honest tiers (written into every report)
* **A** analytic (moon): a real sphere, consistent from every side.
* **A/B** analytic arena / island: real walls, stands, floor; towers and unusual parts are thin relief cards.
* **B** relief / proxy: the reference view is exact, hidden sides are an educated guess.
* **M** mesh-lock: a generated mesh locked to the picture.  The reference view matches within the numbers in the report;
  the sides and back are as good as the supplied mesh (and its texture).
* **C** neural hybrid on your GPU (Hunyuan3D-2.1 / TRELLIS.2 -> `--proxy`): the intended best case; **untested here** (no GPU,
  Hugging Face unreachable from the sandbox).

## Known limits
* Mesh-lock cannot fix a generated mesh that disagrees with the picture (the arena's near towers sit ~40 px off in the
  Meshy example): the front still matches the picture, but its colours land on the wrong surface and show as streaks on
  the floor in side views.  Better registration (non-rigid) or a better mesh is the remedy.
* Untextured generator output (the five Meshy files provided are geometry only) has no side / back colours: they come
  from the diffusion of the visible colours, so they are smooth and muted.  Use a textured export or `--back-image`.
* Thin objects (scythe) reach IoU ~0.91, not 0.99: the fit is rigid; the scythe's thin shaft keeps a pale far side (the echo
  cannot find the near side of a shaft that thin).
* On the arena the near towers' dark picture colours are painted onto the floor behind them (see above) - this is
  worse with the front chart because those floor faces are textured straight from the picture.
* Outline hairline gaps (a pixel) can appear where the picture is anti-aliased against a similar background.

## Next steps
Non-rigid registration of the mesh to the picture; a back-view image generator wired in as a plug-in (any image model that
accepts the front picture); texture inpainting for the diffused areas; skeleton / rig export for characters.

---

(Earlier plan, v5: exact-look)

Goal (from the brief): the asset must look **exactly like the supplied picture** from the picture's own view - shape,
the white details, the colours. The back and sides only have to be plausible; Aura may assume them.

## What was wrong (measured on the scythe, mask recovered from your albedo)
| # | Cause | Effect |
|---|-------|--------|
| 1 | Source-locked mode down-scaled the mask to 1024 px and sampled the boundary every 2 px | 1-3 px chains, wires and tassel tips vanished |
| 2 | Only the largest outer contour was triangulated; a triangle filter rejected 1-2 px features | islands and holes dropped, silhouette IoU 0.933-0.959 |
| 3 | Side walls were built without edge direction, then "repaired" | ~7,000 filled holes, 328k flipped faces, shell still not closed |
| 4 | Texture pixels at the soft anti-aliased edge were replaced by the nearest *fully opaque* pixel (alpha >= 0.98), although the mesh outline sits at alpha 0.5; on a thin object that band is a large share of all pixels | colour PSNR only 14.8 dB (26 dB after the first fix, 52.7 dB once the threshold matched the geometry) |
| 5 | Final QA rendered at half resolution against a Lanczos-filtered reference | misleading 0.987 IoU and a 4x denser retry (403k faces) |
| 6 | The retry re-measured depth from the already-locked shell | wrong depth source |
| 7 | Side walls sampled the rim texel (anti-aliased, pale) | pale streaks on every side view |

## What changed (v5)
- `fidelity.py` rewritten: a vertex on every foreground/background pixel edge (order-free, nothing can be dropped),
  native resolution, render-compare-correct loop, pinch-point splitting (so thin necks that touch stay manifold), side
  walls from *directed* edges (closed by construction, no repair), and wall vertices with UVs inset into the artwork.
- `source_texture()`: native width x height; **every pixel the mesh can show (alpha >= 0.5) is byte-identical to the
  input**; outside pixels take the nearest kept colour so filtering cannot pull in the old background. `--clean-rim`
  (opt-in) swaps the pale 1-px old-background blend for object colour: nicer on dark viewers, but alters real white/silver edges.
- `source_locked` is the default; 2048 px / step 8 / boundary 1; target IoU 0.999; one native-resolution QA pass.
- Dead code removed: unused imports/locals everywhere, 12 unreferenced functions/methods (including the duplicate
  repo-discovery in `backends/_compat.py`), one obsolete test; duplicated `disk` and `vertex_normals` merged;
  `aura_white.raster` no longer imports torch until a Raster is requested; `requirements-dev.txt` added (pytest, trimesh).

## Result (scythe, end to end, CPU, relief geometry, draft preset)
| | v4 report | v5 |
|---|---|---|
| silhouette IoU (reference view) | 0.955 | 0.99995 |
| colour PSNR vs the picture | 14.8 dB | 45.2 dB (52.7 dB before the manifold nudge below) |
| closed shell | no | yes: 0 open edges, 0 over-shared edges after welding by position; `trimesh` reports watertight, consistent winding, positive volume |
| faces | 403k after retry | 111k |

The remaining 0.00005 IoU is a handful of 1-px specks the triangulation cannot reach; the remaining colour error is the
sub-0.5-alpha band the mesh intentionally does not cover.

## Known limits (honest)
1. **Sides and back are assumed**: a thin extruded slab, back = mirrored picture, walls = interior colour stretched
   through the thickness. From an oblique angle it reads as a cut-out and the walls show 1-px staircase banding.
2. Texture seams: wall vertices are duplicates of rim vertices (same position, different UV), so index-level the mesh
   has seam edges; welded by position it is closed. Where thin parts touch (~470 pinch points) each side's vertex copy is
   nudged 0.01 px towards its own part so welding cannot glue the walls back into a 4-face edge (costs ~7 dB PSNR, i.e.
   a handful of rim pixels, and keeps the file valid for Blender/slicers).
3. The mesh is dense (a vertex per pixel edge). Fine for viewers; heavy for a game.
4. White-on-white art depends on the matte: whatever the matte calls background cannot come back.
5. Only the scythe was tested end to end, on a reconstructed input (your original picture was not in the upload).

## Next steps, in order of value
1. **Real back/side geometry while the front stays locked.** TRELLIS.2, Hunyuan3D 2.1 and Pixal3D already have Aura
   backends (2026 roundups still rank TRELLIS.2 as the strongest fully open option for single-image textured meshes).
   Use their mesh only as the depth field for the front and as the shape of the hidden part; texture the hidden side
   from their multi-view output instead of the mirror.
2. **Smooth wall colours:** replace per-vertex inset sampling by a contour-ordered 1-D colour strip appended to the atlas
   (smoothed along the rim) - removes the banding without touching the front UVs.
3. **Mesh diet:** merge collinear boundary vertices, then quadric decimation guarded by a 0.25 px silhouette tolerance
   to reach ~30k faces.
4. **Matte review for white-on-white art:** a "keep the whole bright region connected to the object" mode and a matte
   preview that must be approved before building.
5. **Soft alpha rim:** ship a second material variant with the true soft alpha edge for engines that support it.
6. **CI:** run `tests/` on CPU torch in GitHub Actions; keep the hard-mask fixture (chain, wire, island, hole, pinch).
