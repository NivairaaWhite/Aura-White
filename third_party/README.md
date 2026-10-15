# third_party/

Unpack (or let `aura-white setup` download) the upstream repositories here. Aura White only *reads* their model code.

    third_party/
      Pixal3D-master/          best 1:1 geometry, pixel-aligned to the input image  (~24 GB GPU; --low_vram halves it)
      TRELLIS.2-main/          MIT, O-Voxel sparse latents, 512^3-1536^3 effective  (~24 GB GPU)
      Hunyuan3D-2.1-main/      strong geometry + PBR texture                       (~10 GB GPU)
      InstantMesh-main/        fast geometry + views                               (~8 GB)
      zero123plus-main/        side views for the far side of the object           (~5 GB)
      nvdiffrast-main/         optional faster rasteriser (needs CUDA toolkit + compiler; not required)

You already have the zip files? Point `setup` at them (media files inside are skipped):

    aura-white setup --zip Hunyuan3D-2_1-main.zip InstantMesh-main.zip zero123plus-main.zip nvdiffrast-main.zip
    aura-white setup --repos pixal3d trellis2        # clones from GitHub directly

**Pixal3D and TRELLIS.2 need their own virtualenv** (`bash setup.sh` inside each repo - they pin exact CUDA/torch
versions that would conflict with everything else here). Aura White never imports them directly: it runs
`aura_white/backends/runners/*.py` as a **subprocess** with that environment's own `python`, so a version clash there
can never break Aura White. Point Aura White at the finished environment once:

    export AURA_WHITE_PIXAL3D_DIR=/path/to/Pixal3D-master
    export AURA_WHITE_PIXAL3D_PYTHON=/path/to/Pixal3D-master/.venv/bin/python
    export AURA_WHITE_TRELLIS2_DIR=/path/to/TRELLIS.2-main
    export AURA_WHITE_TRELLIS2_PYTHON=/path/to/TRELLIS.2-main/.venv/bin/python

Then `aura-white studio art.png --geometry pixal3d` (or `--best-of pixal3d trellis2 hunyuan3d`) uses it. `aura-white
doctor` / `aura_white.backends.geometry.status_report()` shows exactly why a backend is unavailable (repo not found,
python not found, not enough free GPU memory) rather than a stack trace.

Any other location works too: `set AURA_WHITE_THIRD_PARTY=D:\models\repos` (Windows) or `export AURA_WHITE_THIRD_PARTY=...`.


## Hunyuan3D-2 (the shape model `aura-white make` uses)

`aura-white setup --hunyuan` downloads `Tencent-Hunyuan/Hunyuan3D-2` into `third_party/Hunyuan3D-2-main/` (or unzip it there by
hand: the folder must contain `hy3dgen/shapegen/pipelines.py`).  Only the **shape** part is used, so nothing is compiled.
Weights (`tencent/Hunyuan3D-2`, `tencent/Hunyuan3D-2mv`) are fetched from Hugging Face on first use (~2.5 GB each).
`aura-white doctor3d --deep` loads it and builds a small mesh to prove the stack works on your machine.
