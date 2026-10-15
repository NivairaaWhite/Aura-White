"""Making Aura smaller and faster WITHOUT guessing: every step here is measured on the frozen benchmark.

  budget            the compute formula: parameters, bytes, FLOPs per sample, GPU-hours for a training run (a calculation)
  block_importance  how much worse is the generator when block i is skipped?  (TinyFusion-style: pruning decided by loss)
  prune_depth       drop the least important DiT blocks, then heal the shorter model by distilling the full one
  distill_guidance  bake classifier-free guidance into the weights: one forward per step instead of two (~2x faster sampling)
  quantize_int8     dynamic int8 weights for the flow's linear layers (CPU inference); a load-time transform, never saved
  measure           parameters, MB, seconds per sample and the benchmark metrics - so a "smaller model" claim has numbers

None of this is allowed to ship on faith: `compare()` in bench.py decides whether the compressed model is still good enough.
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .bench import BenchItem, evaluate
from .config import NativeConfig
from .data import ShardDataset
from .model import AuraNative
from .train import _autocast, cosine_lr, flow_batch, pick_device
from .vae import count_params


# ------------------------------------------------------------------------------------------------ the compute formula
def budget(cfg: NativeConfig, steps: int = 200_000, batch: int = 32, sampling_steps: int = 24, gpu_tflops: float = 120.0,
           utilisation: float = 0.35, bytes_per_param: int = 2) -> Dict:
    """Back-of-the-envelope compute for a preset.  Transformer FLOPs ~ 6 * params * tokens per training sample (fwd + bwd),
    2 * params * tokens per forward.  An honest ESTIMATE: real runs differ by 2x either way; the point is the order of
    magnitude, so the plan is not a guess about whether the hardware exists."""
    m = AuraNative(cfg)
    p = m.params()
    fc, vc = cfg.flow, cfg.vae
    tokens = (vc.grid // fc.patch3) ** 3
    img_tokens = (fc.image_size // fc.patch) ** 2 * fc.max_views
    flow_p = sum(x.numel() for n, x in m.flow.named_parameters() if n.startswith("blocks"))
    attn = 4 * fc.layers * tokens * tokens * fc.d_model          # self-attention score + value mixing, per sample
    fwd = 2 * flow_p * tokens + attn + 2 * fc.layers * tokens * img_tokens * fc.d_model * 2
    train_flops = steps * batch * 3 * fwd                          # fwd + bwd ~ 3x forward
    sample_flops = sampling_steps * 2 * fwd * 2                    # Heun (2 evals) x CFG (2 passes)
    eff = gpu_tflops * 1e12 * utilisation
    return {"preset": cfg.name, "params": p, "weights_mb_fp16": round(p["total"] * bytes_per_param / 2 ** 20, 1),
            "latent_tokens": tokens, "image_tokens": img_tokens, "flow_gflops_per_forward": round(fwd / 1e9, 1),
            "train_flow_gpu_hours": round(train_flops / eff / 3600, 1),
            "sample_gflops_total": round(sample_flops / 1e9, 1), "sample_seconds_est": round(sample_flops / eff, 2),
            "assumptions": {"steps": steps, "batch": batch, "gpu_tflops": gpu_tflops, "utilisation": utilisation,
                            "sampling_steps": sampling_steps, "heun": True, "cfg": True},
            "note": "VAE training excluded (smaller than the flow); data rendering is CPU work and often the real bottleneck."}


# ------------------------------------------------------------------------------------------------ helpers
def _calib(model: AuraNative, ds: ShardDataset, n: int, seed: int, dev):
    rng = np.random.default_rng(seed)
    recs = [ds[int(rng.integers(len(ds)))] for _ in range(n)]
    pts, hc, imgs, vp, vm = flow_batch(recs, model.cfg, rng, dev, 1.0)
    with torch.no_grad():
        mu, logvar = model.vae.encode(pts, hc)
        z = model.vae.sample(mu, logvar)
        z1 = (z - model.mean.view(1, -1, 1, 1, 1)) / model.std.view(1, -1, 1, 1, 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    z0 = torch.randn(z1.shape, generator=g).to(dev)
    t = torch.sigmoid(torch.randn(n, generator=g)).to(dev)
    return z1, z0, t, imgs, vp, vm


@torch.no_grad()
def _flow_loss(flow, calib, blocks: Optional[Sequence[int]] = None) -> float:
    z1, z0, t, imgs, vp, vm = calib
    ctx, mask, pooled = flow.condition(imgs, vp, vm)
    tt = t.view(-1, 1, 1, 1, 1)
    x = (1 - tt) * z0 + tt * z1
    saved = flow.blocks
    if blocks is not None:
        flow.blocks = nn.ModuleList([saved[i] for i in blocks])
    try:
        v = flow(x, t, ctx, mask, pooled)
    finally:
        flow.blocks = saved
    return float(((v - (z1 - z0)) ** 2).mean())


def block_importance(model: AuraNative, ds: ShardDataset, n: int = 32, seed: int = 0, device="cpu") -> List[Dict]:
    """Loss increase when each block alone is skipped (calibration batch is fixed, so the numbers are comparable)."""
    dev = pick_device(device) if isinstance(device, str) else device
    model.to(dev).eval()
    calib = _calib(model, ds, n, seed, dev)
    L = len(model.flow.blocks)
    base = _flow_loss(model.flow, calib)
    out = []
    for i in range(L):
        keep = [j for j in range(L) if j != i]
        out.append({"block": i, "loss_without": _flow_loss(model.flow, calib, keep), "loss_full": base})
    for r in out:
        r["increase"] = r["loss_without"] - r["loss_full"]
    return out


def _train_loop(model: AuraNative, ds: ShardDataset, loss_fn, steps: int, batch: int, lr: float, dev, seed: int,
                stop_after: Optional[float], say) -> Dict:
    flow = model.flow.train()
    for p in model.vae.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    t0, last, done = time.time(), 0.0, 0
    for step in range(steps):
        if stop_after is not None and time.time() - t0 > stop_after:
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr, warmup=min(50, steps // 10 + 1))
        recs = [ds[int(rng.integers(len(ds)))] for _ in range(batch)]
        pts, hc, imgs, vp, vm = flow_batch(recs, model.cfg, rng, dev, 0.8)
        with torch.no_grad():
            mu, logvar = model.vae.encode(pts, hc)
            z = model.vae.sample(mu, logvar)
            z1 = (z - model.mean.view(1, -1, 1, 1, 1)) / model.std.view(1, -1, 1, 1, 1)
        with _autocast(dev):
            loss = loss_fn(z1, imgs, vp, vm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        last, done = float(loss.detach()), step + 1
        if say and (step % 100 == 0 or step == steps - 1):
            say(f"    step {step:4d}/{steps} loss {last:.4f} ({time.time() - t0:.0f}s)")
    flow.eval()
    return {"final_loss": last, "steps_done": done, "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------ depth pruning
def prune_depth(model: AuraNative, ds: ShardDataset, keep: int, heal_steps: int = 300, batch: int = 8, lr: float = 1e-4,
                device="cpu", seed: int = 0, stop_after: Optional[float] = None, say=print) -> Dict:
    """Keep the `keep` most important blocks, then heal by distilling the original.  Returns {model, kept, report}."""
    say = say or (lambda *_: None)
    dev = pick_device(device) if isinstance(device, str) else device
    L = len(model.flow.blocks)
    if not 1 <= keep < L:
        raise ValueError(f"keep must be between 1 and {L - 1}")
    imp = block_importance(model, ds, 24, seed, dev)
    order = sorted(range(L), key=lambda i: -imp[i]["increase"])
    kept = sorted(order[:keep])
    teacher = copy.deepcopy(model.flow).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    s = model.clone().to(dev)
    s.cfg.flow.layers = keep
    s.flow.blocks = nn.ModuleList([s.flow.blocks[i] for i in kept])
    calib = _calib(s, ds, 24, seed + 1, dev)
    before = _flow_loss(s.flow, calib)
    say(f"  pruning {L} -> {keep} blocks (kept {kept}); calibration loss before healing {before:.4f}")

    def loss_fn(z1, imgs, vp, vm):
        B = z1.shape[0]
        ctx, mask, pooled = s.flow.condition(imgs, vp, vm)
        t = torch.sigmoid(torch.randn(B, device=z1.device))
        z0 = torch.randn_like(z1)
        tt = t.view(B, 1, 1, 1, 1)
        x = (1 - tt) * z0 + tt * z1
        v = s.flow(x, t, ctx, mask, pooled)
        with torch.no_grad():
            c2, m2, p2 = teacher.condition(imgs, vp, vm)
            vt = teacher(x, t, c2, m2, p2)
        return 0.5 * ((v - vt) ** 2).mean() + 0.5 * ((v - (z1 - z0)) ** 2).mean()

    rep = _train_loop(s, ds, loss_fn, heal_steps, batch, lr, dev, seed, stop_after, say)
    after = _flow_loss(s.flow, calib)
    s.card.setdefault("history", []).append({"op": "prune_depth", "from_layers": L, "to_layers": keep, "kept": kept,
                                             "calib_loss_before_heal": before, "calib_loss_after_heal": after, **rep})
    return {"model": s.eval(), "kept": kept, "importance": imp,
            "report": {"calib_loss_before_heal": before, "calib_loss_after_heal": after, **rep}}


# ------------------------------------------------------------------------------------------------ guidance distillation
def distill_guidance(model: AuraNative, ds: ShardDataset, steps: int = 400, batch: int = 8, lr: float = 1e-4,
                     w_range=(1.0, 6.0), device="cpu", seed: int = 0, stop_after: Optional[float] = None, say=print) -> Dict:
    """Teach the flow to output   v_u + w (v_c - v_u)   in one pass, with w as an input.  Afterwards sampling needs half the
    forward passes.  The frozen teacher is a copy of the model's own flow."""
    dev = pick_device(device) if isinstance(device, str) else device
    s = model.clone().to(dev)
    teacher = copy.deepcopy(s.flow).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    def loss_fn(z1, imgs, vp, vm):
        B = z1.shape[0]
        t = torch.sigmoid(torch.randn(B, device=z1.device))
        z0 = torch.randn_like(z1)
        tt = t.view(B, 1, 1, 1, 1)
        x = (1 - tt) * z0 + tt * z1
        w = torch.empty(B, device=z1.device).uniform_(*w_range)
        with torch.no_grad():
            c, m, p = teacher.condition(imgs, vp, vm)
            cu, mu_, pu = teacher.condition(imgs, vp, vm, torch.ones(B, dtype=torch.bool, device=z1.device))
            vc, vu = teacher(x, t, c, m, p), teacher(x, t, cu, mu_, pu)
            tgt = vu + w.view(B, 1, 1, 1, 1) * (vc - vu)
        ctx, mask, pooled = s.flow.condition(imgs, vp, vm)
        return ((s.flow(x, t, ctx, mask, pooled, w) - tgt) ** 2).mean()

    rep = _train_loop(s, ds, loss_fn, steps, batch, lr, dev, seed, stop_after, say)
    s.guidance_distilled = True
    s.card.setdefault("history", []).append({"op": "distill_guidance", "w_range": list(w_range), **rep})
    return {"model": s.eval(), "report": rep}


# ------------------------------------------------------------------------------------------------ int8
def quantize_int8(model: AuraNative) -> AuraNative:
    """A copy whose flow linear layers hold int8 weights (dynamic activation quantisation; CPU only).  In-memory only: the
    checkpoint format stays plain fp32/fp16 tensors, so nothing here can break a load."""
    m = model.clone().to("cpu").eval()
    m.flow = torch.ao.quantization.quantize_dynamic(m.flow, {nn.Linear}, dtype=torch.qint8)
    m.card.setdefault("history", []).append({"op": "quantize_int8"})
    return m


# ------------------------------------------------------------------------------------------------ measurement
def measure(model: AuraNative, items: Sequence[BenchItem], seeds=(0, 1), steps: int = 16, cfg_scale: float = 3.0,
            res: int = 40, say=None) -> Dict:
    """Size, speed and quality in one record - the only way a compression claim is allowed to be made."""
    n_flow = sum(p.numel() for p in model.flow.parameters())
    n_vae = count_params(model.vae)
    t0 = time.time()
    r = evaluate(model, items, seeds=seeds, steps=steps, cfg_scale=cfg_scale, res=res, say=say)
    n = max(1, len(items) * len(seeds))
    return {"params_flow": int(n_flow), "params_vae": int(n_vae), "mb_fp32": round((n_flow + n_vae) * 4 / 2 ** 20, 2),
            "sec_per_sample": round((time.time() - t0) / n, 3), "layers": len(model.flow.blocks),
            "guidance_distilled": bool(model.guidance_distilled), "summary": r["summary"]}
