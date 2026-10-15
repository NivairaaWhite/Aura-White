#!/usr/bin/env python
"""Run TRELLIS.2 on one image and write a GLB.  Executed with the Python of the TRELLIS.2 environment:

    python trellis2_run.py --repo /path/to/TRELLIS.2 --image in.png --output out.glb [--seed 42] [--simplify N]

Follows the usage in the TRELLIS.2 README (Trellis2ImageTo3DPipeline + o_voxel.postprocess.to_glb).  Not imported
by Aura White, so a dependency conflict in that environment can never break the rest of the stack."""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--simplify", type=int, default=1_000_000, help="decimation target of the exported GLB")
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--pipeline-type", default="1024_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--max-tokens", type=int, default=49152)
    a = ap.parse_args()

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, a.repo)
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    torch.manual_seed(a.seed)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(a.model)
    pipeline.cuda()
    image = Image.open(a.image)
    # an RGBA picture with a real alpha channel is used as is (no background-removal model is needed)
    mesh = pipeline.run(image, seed=a.seed, pipeline_type=a.pipeline_type, max_num_tokens=a.max_tokens)[0]
    mesh.simplify(16777216)                                    # nvdiffrast limit (README)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=a.simplify, texture_size=a.texture_size, remesh=True, remesh_band=1, remesh_project=0,
        verbose=True)
    glb.export(a.output)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
