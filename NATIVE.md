# Aura Native - making Aura its own model (v8)

This document answers one question honestly: **can Aura stop depending on third-party generators and become a model of its
own, trained from other models' refined data, improving itself, small enough to run anywhere?**

Short answer: the machinery is built, runs, and is measured.  What it cannot do yet is match the big open models, because that
takes real data and GPU-days that this repository cannot contain.  Nothing below pretends otherwise; where a number is a
measurement it says so, where it is an estimate it says that.

## 1. What exists (all in `aura_white/native/`, numpy + PyTorch, no compiled extension)

| piece | file | what it does |
|---|---|---|
| licence ledger | `licenses.py` | who may teach Aura. TRELLIS.2, TripoSG, TripoSR (MIT) yes; **Hunyuan3D (any version or spelling of the name) never** (its licence §5(b) forbids using output to improve another AI model, and excludes the EU/UK/South Korea); Zero123++ weights (CC-BY-NC) only in research mode; anything unverified is denied by default. Every shard and checkpoint carries the decision. |
| procedural factory | `procedural.py` | unlimited assets with an **exact** signed distance and a front that differs from the back. Ground truth you can trust, free. |
| own shape VAE | `vae.py` | coloured surface points -> a structured latent grid -> signed distance + colour at any point (the TRELLIS family of idea, own implementation). |
| own generator | `flow.py`, `model.py` | rectified-flow transformer over that latent grid, conditioned on 1-4 pictures (own ViT encoder, azimuth/elevation embeddings). It **samples**, so a hidden back is drawn, not averaged or mirrored; a back picture just adds tokens. |
| quality gates | `gates.py` | closed mesh, real depth (not a slab, no stair-steps), a back that is not a copy of the front, sane colours, agrees with the picture, valid GLB. No ground truth needed. |
| frozen benchmark | `bench.py` | same items, same seeds, every round. **Oracle** numbers (chamfer, F-score, side IoU, back colour error) are used only to *judge*, never to pick training samples. |
| teachers | `teach.py` | any open generator, run by you (`CommandTeacher`), kept only if its licence allows it **and** the mesh passes the gates. |
| self-improvement | `evolve.py` | generate K candidates -> verifier keeps winners -> fine-tune with anchor data mixed in -> promote only if the frozen benchmark improves and diversity holds. |
| compression | `compress.py` | compute budget formula; block importance; depth pruning + distillation; guidance distillation; int8. Every step is measured. |
| sheet reconstruction | `multiview.py` | front/back/left/right pictures -> textured GLB, **no network, no GPU**; or paint a mesh from any generator. See section 5. |
| commands | `cli.py` | `aura-white native ...` and `aura-white sheet ...` |

`aura-white native licences` prints the ledger; `aura-white native budget` prints the compute plan.

## 2. What was measured (CPU, the `tiny` preset: 2.9 M parameters, procedural data)

These prove the **mechanism**. They say nothing about quality on real art.

* Shape VAE, held-out procedural assets: volume IoU 0.848, chamfer 0.0207 (unit-box units).  The first design (attention over a
  point set) reached volume IoU 0.27 and was replaced by the structured grid.
* Generator, frozen benchmark (60 held-out assets x 3 seeds): chamfer 0.063 +/- 0.005, F-score@0.06 0.75, verifier pass rate 0.48.
  Samples are real 3-D with a distinct back; colour is muddy and the "prop" family still fails.
* Gates: they pass real procedural meshes and fail a v7-style slab. They also show that **front fidelity alone is gameable**
  (a slab matches the front perfectly), which is why the depth and back gates exist.

### Does self-improvement work?  Not demonstrated.

One round, four arms from the same checkpoint, same 480 candidates (mean verifier score 0.58; 56 of 120 pictures had a winner above 0.75):

| arm | oracle chamfer (lower = better) | verifier score | pass rate |
|---|---|---|---|
| base checkpoint | 0.0627 +/- 0.0049 | 0.643 | 0.478 |
| verifier-selected self data + anchor | 0.0709 | 0.628 | 0.456 |
| random-selected self data + anchor | 0.0665 | 0.605 | 0.406 |
| anchor data only (no self data) | 0.0639 | 0.662 | 0.517 |
| verifier-selected, no anchor | 0.0656 | 0.612 | 0.422 |

The experiment forced promotion (`always_promote`) so that all four arms could be compared side by side; the real rule would have
rejected the verifier arm ("chamfer did not improve beyond 0 SE": -0.0083 against a standard error of 0.0076, wrong direction).
Everything is within about one standard error of the base; **no arm beat the base on ground truth**. The verifier arm beat the
random arm on the verifier's own score, not on correctness. That is what the theory predicts (a self-trained model can gain only
as far as its verifier's knowledge reaches, and training on its own raw samples risks collapse), so the loop ships with
the safeguards (anchor data, frozen oracle gate, diversity floor) and **no claim that it improves a model**.  One round, one seed:
read it as "no evidence", not "evidence of absence".  The reliable lever remains more real data (and teacher data).

### Compression (16 benchmark items x 1 seed, so differences under ~0.007 chamfer are noise)

| variant | chamfer | verifier score | seconds / sample (incl. decode + verify) |
|---|---|---|---|
| base | 0.0613 | 0.661 | 0.82 |
| guidance distilled (60 steps) | 0.0629 | 0.682 | **0.57** |
| int8 flow | 0.0617 | 0.685 | 0.90 (no gain: the flow is a small share of the time at this size) |
| depth-pruned 4 -> 3 blocks, 60 healing steps | 0.0686 | 0.582 | 0.70 (quality lost; needs far longer healing) |

Guidance distillation is the one that paid off here (one forward per step instead of two).  Pruning and int8 matter at the
large presets, where the flow dominates the time; they are implemented and tested, not proven at that scale.

## 3. Compute budget (formula in `compress.budget`; an ESTIMATE, +/- 2x)

| preset | parameters | fp16 weights | latent tokens | flow GFLOPs / forward | training (200k steps, batch 32, 120 TFLOPs at 35%) |
|---|---|---|---|---|---|
| tiny | 2.9 M | 6 MB | 64 | 0.3 | minutes (what this repository ran) |
| s | 98 M | 187 MB | 512 | 100 | ~13 GPU-hours |
| m | 292 M | 557 MB | 1,728 | 1,050 | ~133 GPU-hours |
| l | 761 M | 1.45 GB | 4,096 | 7,220 | ~920 GPU-hours |

Training the VAE is cheaper than the flow and excluded.  Rendering training pictures is CPU work and is often the real bottleneck.
**The `s` preset on a single rented GPU for a day or two is the realistic first target, and only for a narrow domain**
(game props, weapons, characters in one style).  A general model competing with TRELLIS.2 or Hunyuan needs orders of magnitude
more data and compute than that.

## 4. The path to real quality (what to run on a GPU)

```
aura-white native licences                                  # who you may learn from
aura-white native make-data data/proc -n 20000 --workers 8  # exact-ground-truth procedural base
aura-white native ingest my_meshes data/real --default-licence own          # your own assets (or Objaverse with --licences tags.json)
aura-white native teach pictures/ data/teach --teacher trellis2 --command "python run_trellis2.py --image {image} --out {out}"
aura-white native train-vae data/real --preset s --ckpt s.pt --steps 100000 --device cuda
aura-white native train-flow data/real --preset s --ckpt s.pt --steps 200000 --device cuda --resume
aura-white native bench s.pt data/proc -n 120 --out bench.json
aura-white native evolve data/real data/pool --ckpt s.pt --rounds 3     # promotes only on the frozen benchmark
aura-white native distill s.pt data/real --out s_fast.pt                 # ~2x faster sampling
aura-white native approve s_fast.pt --bench bench.json --data data/real data/teach   # refuses demo / procedural-only models
AURA_NATIVE_CKPT=s_fast.pt aura-white make --allow-basic knight.png     # the checkpoint now competes as a back-end
```

Rules the code enforces: teachers need a licence the ledger allows (a name containing "hunyuan" always resolves to the denied entry, and a command that mentions it is refused - this catches accidents, it cannot stop someone who relabels a model on purpose); a teacher's mesh must pass the gates and match its own
picture; `approve` needs >= 5,000 real (non-procedural) assets, a passing benchmark and commercially clean data;
`AuraNativeShape` ignores a checkpoint that is not approved unless you set `AURA_NATIVE_ALLOW_UNAPPROVED=1`.  Checkpoints are
plain tensors loaded with `weights_only=True`.

Things deliberately NOT done: no pretrained weights are bundled; no third-party model is called at inference; the self-improvement
loop is not allowed to use ground truth to pick samples; reflow / multi-step distillation is not implemented (guidance distillation is).

## 5. Sheet reconstruction: `aura-white sheet` (no GPU, no weights)

If the artist already drew the other sides, nothing needs to be guessed:

```
aura-white sheet --front front.png --back back.png --left left.png --right right.png --height 1.68 --name aura
aura-white sheet --mesh generator_output.glb --front front.png --back back.png --left left.png --right right.png   # keep the mesh, repaint it
```

Pictures are matted (soft outline, watermarks dropped), registered into one orthographic frame (back is checked against the
mirrored front, right against the mirrored left), carved into a closed mesh whose cross-sections are rounded (not boxes), decimated,
unwrapped (xatlas) and painted texel by texel from the pictures that really see each spot (weight = how squarely it faces the camera,
zero when occluded).  The back comes from the back picture, the sides from the side pictures.  Output: GLB, albedo, turntable
preview, a JSON report with the measured agreement.

Measured (procedural figure with known ground truth, four views in): all four outlines reproduced at IoU 0.98-0.99; chamfer to the
true shape 0.009 of the height (F-score@2% 0.94); the unseen sides come out right (the back's stripes are there).  Painting a
supplied mesh: the tool found the correct quarter-turn and shift of a scrambled mesh (outline agreement 0.69 -> 0.99) and
reproduced its colours with mean error 0.096 versus 0.184 for a flat-colour guess.

### The limit, stated plainly

Four outlines cannot show what hides *inside* them.  Carving is exact in those four views and a rounded interpolation in between.
For props, weapons, buildings and slim bodies that is good.  For **loose white clothing and very long hair** (the character this
was tried on) the sleeves, hair and torso merge into one silhouette, so the carve is a smooth bell-shaped blockout: real depth, a real
back, correct painting from each side - but not a finished character.  That is a property of the input, not a bug to tune away.
Three honest routes to a finished model: (1) give 45-degree views as well; (2) use a generator for the shape and let `--mesh` paint
it - for four-view sheets Hunyuan3D-2mv (the `make` back-end in v7) is built for exactly this input; (3) take the blockout into a
sculpting tool.  Route (2) was not run here (no GPU or weights in the build sandbox).

## 6. Limits of the whole v8 effort

* The trained `tiny` model is a demo of the mechanism.  Do not expect it to generate real characters.
* Hunyuan3D remains usable by *you* as a v7 back-end for generating assets (mind its territory limits); it is just never a *teacher*.
* The licence ledger records what was read on 2026-10-13; licences change - re-read before commercial use.  It is bookkeeping, not legal advice.
* One benchmark, one seed per arm, one round: no statistical claim is made beyond the standard errors printed.
