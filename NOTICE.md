# Notices

Aura White re-implements, in plain PyTorch, the network architecture published as **TripoSR**
(Tochilkin et al., Tripo AI and Stability AI, 2024; https://github.com/VAST-AI-Research/TripoSR,
MIT licence). The pretrained weights it loads are the official `stabilityai/TripoSR` checkpoint
(MIT licence) and are **not** included in this package - `aura-white fetch` obtains them (or re-uses
a copy you already have).

The transformer block layout follows the Hugging Face `diffusers` design (Apache-2.0) and the image
encoder is the DINO ViT-B/16 architecture (Apache-2.0, Meta AI); both are written from scratch here so
that no `diffusers` / `transformers` install is needed.

Keep these credits if you redistribute the weights or derived work.

## Textured pipeline (`aura-white studio`)

The studio orchestrates - it does **not** bundle - these third-party projects. You unpack or download them yourself
(`aura-white setup`), and their licences apply to what you generate with them:

* **Hunyuan3D-2.1** (Tencent) - Tencent Hunyuan 3D 2.1 Community License (territory and usage conditions apply; read it
  before commercial use). https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
* **InstantMesh** (Tencent ARC) - code Apache-2.0; weights Apache-2.0. https://github.com/TencentARC/InstantMesh
* **Zero123++** (SUDO-AI) - code Apache-2.0; **weights CC-BY-NC 4.0 (non-commercial)**. https://github.com/SUDO-AI-3D/zero123plus
  Textures from Zero123++ side views inherit that restriction. Run with `--multiview off` to avoid it.
* **nvdiffrast** (NVIDIA) - NVIDIA Source Code License (non-commercial research use). Optional; Aura White's own
  PyTorch rasteriser is the default.
* **xatlas** (via xatlas-python, MIT) and **fast-simplification** (MIT): optional, each with a built-in fallback.


## Aura White 7 - models the true-3D engine can load (none is bundled; each is fetched on first use)

| model | used for | licence (check the model card before commercial use) |
|---|---|---|
| Hunyuan3D-2 / 2mv shape (Tencent) | `make`: shape from one picture / several views | Tencent Hunyuan Community License - it **excludes some territories** (the EU, UK and South Korea at the time of writing) and has a large-service threshold; read the LICENSE file in the model repo |
| TripoSR (Stability AI / Tripo) | built-in network, `--allow-basic` fall-back | MIT |
| Stable Diffusion XL base | text -> reference picture; painting unseen sides | CreativeML Open RAIL++-M |
| SDXL depth ControlNet (`diffusers/controlnet-depth-sdxl-1.0`) | painting unseen sides | OpenRAIL++ |
| IP-Adapter (`h94/IP-Adapter`) | carrying the reference's colours to painted sides | Apache-2.0 |
| rembg segmenters (isnet-anime, isnet-general-use, u2net, birefnet) | outlines | Apache-2.0 / MIT (per model; `bria-rmbg` is non-commercial and is never selected) |
| OpenCV GrabCut | outline edge refinement | Apache-2.0 |

`aura-white make --tier max` additionally allows Pixal3D, TRELLIS.2 and Hunyuan3D-2.1 through the existing back-ends
(see `third_party/README.md` for their licences).


## Aura White 8 - Aura Native and the sheet tools

* No third-party model weights are bundled or called by `aura_white.native` at inference.  Optional libraries: `xatlas` (MIT),
  `fast-simplification` (MIT), `scikit-image` (BSD), `opencv-python-headless` (Apache-2.0), `trimesh` (MIT), `scipy` (BSD).
* **Teacher licences** (recorded in `aura_white/native/licenses.py`, read 2026-10-13; re-read before relying on them):
  TRELLIS.2 (MIT), TripoSG (MIT) and TripoSR (MIT) may be used as teachers.  **Hunyuan3D 2.0 / 2.1 may NOT be a teacher**: the
  Tencent Hunyuan 3D Community Licenses (section 5(b)) forbid using the Works or any Output to improve another AI model, and
  exclude the EU, UK and South Korea.  Aura refuses such a teacher in code and gives no override.  Using Hunyuan3D yourself as the
  v7 `make` back-end to generate assets is a separate matter governed by its licence.  Zero123++ weights are CC-BY-NC: research mode only.
* Objaverse-style datasets: per-asset licences are filtered (CC0 / CC-BY in commercial mode), not assumed.
* This is engineering bookkeeping, not legal advice.
