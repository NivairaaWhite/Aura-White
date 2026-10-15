"""Numerical parity check: Aura White vs the original TripoSR source code.

Tiny mode (default, no downloads, ~seconds):
    python tools/parity_check.py --triposr C:\\path\\to\\TripoSR-main
Real mode (needs the official files config.yaml + model.ckpt in one folder, and the original repo's
old dependencies working - the point is to prove Aura White reproduces it):
    python tools/parity_check.py --triposr C:\\path\\to\\TripoSR-main --real C:\\path\\to\\snapshot_dir --image chair.png

Everything the original needs that is broken/unavailable (torchmcubes, rembg, xatlas, moderngl) is
stubbed - none of it is used by the parts being compared.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch


def _stub(name: str):
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        m = types.ModuleType(name)
        m.marching_cubes = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))
        m.remove = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))
        m.new_session = lambda *a, **k: None
        sys.modules[name] = m


def _randomize(sd: dict, seed: int = 0) -> dict:
    """Well-conditioned random weights (signals neither vanish nor explode), so that a wrong layer
    anywhere changes the output measurably."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sd.items():
        if not v.is_floating_point():
            out[k] = v
        elif k.endswith(("cls_token", "position_embeddings", "tokenizer.embeddings")):
            out[k] = 0.5 * torch.randn(v.shape, generator=g)
        elif v.ndim == 1 and k.endswith("weight") and "norm" in k:
            out[k] = 1 + 0.1 * torch.randn(v.shape, generator=g)
        elif v.ndim >= 2:
            fan_in = max(1, int(np.prod(v.shape[1:])))
            gain = 0.85 if k.startswith("decoder.") else 1.6
            out[k] = (gain / fan_in ** 0.5) * torch.randn(v.shape, generator=g)
        else:
            out[k] = 0.1 * torch.randn(v.shape, generator=g)
    return out


def _report(name: str, a: torch.Tensor, b: torch.Tensor, tol: float) -> bool:
    a, b = a.float().cpu(), b.float().cpu()
    err = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), 1e-6)
    print(f"        [{name}: magnitude {scale:.3f}, std {a.std().item():.4f}]")
    ok = err <= tol * scale
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<34} max|diff|={err:.3e}  (rel {err / scale:.2e}, tol {tol:.0e})")
    return ok


def tiny(triposr: Path, cond_size: int) -> bool:
    from omegaconf import OmegaConf
    from transformers import ViTConfig

    from aura_white.config import AuraConfig
    from aura_white.model import AuraNet
    from aura_white.model.renderer import spherical_cameras, render_rays

    import tsr.models.tokenizers.image as tok_image
    from tsr.system import TSR

    ac = AuraConfig.tiny()
    ac.cond_image_size = cond_size
    tmp = Path(tempfile.mkdtemp())
    vit_cfg = ViTConfig(hidden_size=ac.enc_hidden, num_hidden_layers=ac.enc_layers, num_attention_heads=ac.enc_heads,
                        intermediate_size=ac.enc_mlp, image_size=ac.enc_pos_grid * ac.enc_patch, patch_size=ac.enc_patch,
                        qkv_bias=True, layer_norm_eps=ac.enc_eps)
    vit_cfg.save_pretrained(tmp)
    tok_image.hf_hub_download = lambda repo_id, filename: str(tmp / "config.json")

    cfg = OmegaConf.create({
        "cond_image_size": cond_size,
        "image_tokenizer_cls": "tsr.models.tokenizers.image.DINOSingleImageTokenizer",
        "image_tokenizer": {"pretrained_model_name_or_path": "local"},
        "tokenizer_cls": "tsr.models.tokenizers.triplane.Triplane1DTokenizer",
        "tokenizer": {"plane_size": ac.plane_size, "num_channels": ac.plane_channels},
        "backbone_cls": "tsr.models.transformer.transformer_1d.Transformer1D",
        "backbone": {"in_channels": ac.plane_channels, "num_attention_heads": ac.bb_heads, "attention_head_dim": ac.bb_head_dim,
                     "num_layers": ac.bb_layers, "cross_attention_dim": ac.enc_hidden, "norm_num_groups": ac.bb_norm_groups},
        "post_processor_cls": "tsr.models.network_utils.TriplaneUpsampleNetwork",
        "post_processor": {"in_channels": ac.plane_channels, "out_channels": ac.triplane_out_channels},
        "decoder_cls": "tsr.models.network_utils.NeRFMLP",
        "decoder": {"in_channels": 3 * ac.triplane_out_channels, "n_neurons": ac.mlp_neurons,
                    "n_hidden_layers": ac.mlp_hidden_layers, "activation": "silu"},
        "renderer_cls": "tsr.models.nerf_renderer.TriplaneNeRFRenderer",
        "renderer": {"radius": ac.radius, "feature_reduction": "concat", "density_activation": "exp",
                     "density_bias": ac.density_bias, "num_samples_per_ray": ac.samples_per_ray},
    })
    orig = TSR(cfg).eval()
    orig.load_state_dict(_randomize(orig.state_dict()))
    mine = AuraNet(ac).eval()

    from aura_white.weights import normalize_state_dict

    osd = orig.state_dict()
    res = mine.load_state_dict(normalize_state_dict(osd), strict=False)
    keys_ok = not res.missing_keys and not res.unexpected_keys
    renamed = any(".layers." in k for k in osd)
    print(f"  {'OK  ' if keys_ok else 'FAIL'} checkpoint keys/shapes map 1:1 (installed transformers uses "
          f"{'NEW' if renamed else 'old'} ViT names)")
    ok = keys_ok

    rng = np.random.RandomState(1)
    img = rng.rand(80, 100, 3).astype(np.float32)
    with torch.no_grad():
        codes_o = orig([img], device="cpu")
        x = torch.from_numpy(img).permute(2, 0, 1)[None]
        codes_m = mine.encode(x)
        ok &= _report("scene codes (encoder+backbone)", codes_o, codes_m, 2e-4)

        pts = (torch.rand(500, 3, generator=torch.Generator().manual_seed(3)) * 2 - 1) * ac.radius
        q = orig.renderer.query_triplane(orig.decoder, pts, codes_o[0])
        d, c = mine.query(codes_m[0], pts)
        ok &= _report("density_act at random points", q["density_act"][:, 0], d, 2e-4)
        ok &= _report("color at random points", q["color"], c, 2e-4)

        imgs_o = orig.render(codes_o, n_views=4, height=24, width=24, return_type="pt")[0]
        ro, rd = spherical_cameras(4, 0.0, 1.9, 40.0, 24, 24)
        imgs_m = [render_rays(ac, mine.decoder, codes_m[0], ro[i].reshape(-1, 3), rd[i].reshape(-1, 3)).reshape(24, 24, 3) for i in range(4)]
        ok &= _report("rendered views (4x24x24)", torch.stack(imgs_o), torch.stack(imgs_m), 5e-4)

        # canary: the comparison above must be able to *detect* an error in every stage
        import copy

        def bump(net, getter):
            n2 = copy.deepcopy(net)
            with torch.no_grad():
                getter(n2).add_(0.05)
            return n2

        canaries = {
            "encoder": lambda n: n.image_tokenizer.model.encoder.layer[-1].attention.attention.key.weight,
            "backbone": lambda n: n.backbone.transformer_blocks[-1].attn2.to_v.weight,
            "decoder": lambda n: n.decoder.layers[0].weight,
        }
        for name, getter in canaries.items():
            n2 = bump(mine, getter)
            cm2 = n2.encode(x)
            dd = n2.query(codes_m[0], pts)[0]
            moved = max((codes_o - cm2).abs().max().item() / max(codes_o.abs().max().item(), 1e-6),
                        (q["density_act"][:, 0] - dd).abs().max().item() / max(q["density_act"].abs().max().item(), 1e-6))
            sens = moved > 20 * 2e-4
            print(f"  {'OK  ' if sens else 'FAIL'} canary: perturbing the {name} is detected (rel change {moved:.1e})")
            ok &= sens
    return ok


def full_shapes(triposr: Path) -> bool:
    """Build BOTH networks at the real TripoSR size on the 'meta' device (no memory used) and require
    identical parameter names and shapes - so the official checkpoint will load into Aura White."""
    from omegaconf import OmegaConf
    from transformers import ViTConfig

    from aura_white.config import AuraConfig
    from aura_white.model import AuraNet
    from aura_white.weights import normalize_state_dict

    import tsr.models.tokenizers.image as tok_image
    from tsr.system import TSR

    tmp = Path(tempfile.mkdtemp())
    ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, image_size=224,
              patch_size=16, qkv_bias=True, layer_norm_eps=1e-12).save_pretrained(tmp)
    tok_image.hf_hub_download = lambda repo_id, filename: str(tmp / "config.json")
    cfg = OmegaConf.create({   # = stabilityai/TripoSR config.yaml
        "cond_image_size": 512,
        "image_tokenizer_cls": "tsr.models.tokenizers.image.DINOSingleImageTokenizer",
        "image_tokenizer": {"pretrained_model_name_or_path": "facebook/dino-vitb16"},
        "tokenizer_cls": "tsr.models.tokenizers.triplane.Triplane1DTokenizer",
        "tokenizer": {"plane_size": 32, "num_channels": 1024},
        "backbone_cls": "tsr.models.transformer.transformer_1d.Transformer1D",
        "backbone": {"in_channels": 1024, "num_attention_heads": 16, "attention_head_dim": 64, "num_layers": 16,
                     "cross_attention_dim": 768},
        "post_processor_cls": "tsr.models.network_utils.TriplaneUpsampleNetwork",
        "post_processor": {"in_channels": 1024, "out_channels": 40},
        "decoder_cls": "tsr.models.network_utils.NeRFMLP",
        "decoder": {"in_channels": 120, "n_neurons": 64, "n_hidden_layers": 9, "activation": "silu"},
        "renderer_cls": "tsr.models.nerf_renderer.TriplaneNeRFRenderer",
        "renderer": {"radius": 0.87, "feature_reduction": "concat", "density_activation": "exp",
                     "density_bias": -1.0, "num_samples_per_ray": 128},
    })
    with torch.device("meta"):
        orig = TSR(cfg)
        mine = AuraNet(AuraConfig())
    o = {k: tuple(v.shape) for k, v in normalize_state_dict(orig.state_dict()).items()}
    m = {k: tuple(v.shape) for k, v in mine.state_dict().items()}
    n_o = sum(int(np.prod(v.shape)) for k, v in normalize_state_dict(orig.state_dict()).items())
    n_m = sum(int(np.prod(v.shape)) for v in mine.state_dict().values())
    ok = o == m
    print(f"  {'OK  ' if ok else 'FAIL'} full-size architecture: {len(m)} tensors, {n_m / 1e6:.1f} M parameters "
          f"(original: {len(o)} tensors, {n_o / 1e6:.1f} M) - names and shapes {'identical' if ok else 'DIFFER'}")
    if not ok:
        print("   only in original:", sorted(set(o) - set(m))[:5], " only in Aura White:", sorted(set(m) - set(o))[:5],
              " shape mismatches:", [k for k in o if k in m and o[k] != m[k]][:5])
    return ok


def real(triposr: Path, snapshot: Path, image: Path) -> bool:
    from PIL import Image
    from tsr.system import TSR

    from aura_white.model import AuraNet
    from aura_white.config import AuraConfig
    from aura_white._compat import load_state_dict_file

    orig = TSR.from_pretrained(str(snapshot), config_name="config.yaml", weight_name="model.ckpt").eval()
    mine = AuraNet(AuraConfig()).eval()
    from aura_white.weights import normalize_state_dict

    mine.load_state_dict(normalize_state_dict(load_state_dict_file(snapshot / "model.ckpt")))
    im = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / 255.0
    with torch.no_grad():
        co = orig([im], device="cpu")
        cm = mine.encode(torch.from_numpy(im).permute(2, 0, 1)[None])
    return _report("scene codes with REAL weights", co, cm, 1e-3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triposr", required=True, help="path to the original TripoSR-main folder")
    ap.add_argument("--real", help="folder containing the official config.yaml and model.ckpt")
    ap.add_argument("--image", help="image for --real mode")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.triposr).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    for m in ("torchmcubes", "rembg", "xatlas", "moderngl"):
        _stub(m)
    ok = True
    print("full-size shapes (meta device)")
    ok &= full_shapes(Path(a.triposr))
    if a.real:
        ok = real(Path(a.triposr), Path(a.real), Path(a.image))
    else:
        for size in (64, 96):  # 64: native position grid; 96: exercises bicubic position-embedding resize
            print(f"tiny model, cond_image_size={size}")
            ok &= tiny(Path(a.triposr), size)
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
