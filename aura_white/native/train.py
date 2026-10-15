"""Training loops: the shape VAE, the latent flow generator, and the shared plumbing (batches, EMA, schedule, logging).

Both loops are resumable, log JSON lines, keep an EMA copy of the weights (that copy is what is saved and sampled from),
run in bf16 autocast on CUDA and plain fp32 on a CPU.  Data comes from `data.ShardDataset`.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import ShardDataset
from .model import AuraNative


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:
    def __init__(self, module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        d = self.decay
        for k, v in module.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, module: torch.nn.Module):
        module.load_state_dict(self.shadow)


def cosine_lr(step: int, total: int, base: float, warmup: int = 100, floor: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def rot_y_batch(az_deg: torch.Tensor) -> torch.Tensor:
    """(B,) degrees -> (B,3,3) matrices that turn the WORLD so a camera at azimuth `az` ends up on +z (the first-view frame)."""
    a = -torch.deg2rad(az_deg)
    c, s = a.cos(), a.sin()
    R = torch.zeros(az_deg.shape[0], 3, 3, device=az_deg.device)
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1.0, -s, c
    return R


def _rot_pts(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return x @ R.transpose(1, 2)


# ------------------------------------------------------------------------------------------------ batches
def _take(a: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(a), size=n, replace=len(a) < n)
    return idx


def vae_batch(recs, cfg, rng: np.random.Generator, device, yaw: float = 0.0):
    pts, q, sdf, rgb, hc = [], [], [], [], []
    for r in recs:
        ip = _take(r["pts"], cfg.n_points, rng)
        iq = _take(r["q"], cfg.n_queries, rng)
        pts.append(r["pts"][ip].astype(np.float32))
        q.append(r["q"][iq].astype(np.float32))
        sdf.append(r["sdf"][iq].astype(np.float32))
        rgb.append(r["qrgb"][iq].astype(np.float32) / 255.0)
        hc.append(float(r["has_colour"]))
    t = lambda a: torch.from_numpy(np.stack(a)).to(device)
    pts, q, sdf, rgb = t(pts), t(q), t(sdf), t(rgb)
    hc = torch.tensor(hc, device=device)
    if yaw > 0:
        az = (torch.rand(len(recs), device=device) * 2 - 1) * yaw
        az = torch.where(torch.rand(len(recs), device=device) < 0.5, torch.zeros_like(az), az)
        R = rot_y_batch(-az)
        pts = torch.cat([_rot_pts(pts[..., :3], R), _rot_pts(pts[..., 3:6], R), pts[..., 6:]], -1)
        q = _rot_pts(q, R)
    return pts, q, sdf, rgb, hc


def flow_batch(recs, mcfg, rng: np.random.Generator, device, front_prob: float = 0.8, extra_view_probs=(0.55, 0.2, 0.15, 0.10)):
    """First view = the frame.  Returns (pts rotated into that frame, has_colour, imgs, vparams, vmask)."""
    fc, vc = mcfg.flow, mcfg.vae
    B, V, S = len(recs), fc.max_views, fc.image_size
    imgs = np.zeros((B, V, 4, S, S), np.float32)
    vp = np.zeros((B, V, 2), np.float32)
    vm = np.zeros((B, V), bool)
    first_az = np.zeros(B, np.float32)
    pts, hc = [], []
    for b, r in enumerate(recs):
        az, el = r["view_az"], r["view_el"]
        nv = len(az)
        front = np.flatnonzero((np.abs(az) < 1e-3) & (np.abs(el) < 1e-3))
        i1 = int(front[0]) if (len(front) and rng.random() < front_prob) else int(rng.integers(nv))
        k = int(rng.choice(len(extra_view_probs), p=np.asarray(extra_view_probs) / np.sum(extra_view_probs)))
        others = [j for j in range(nv) if j != i1]
        rng.shuffle(others)
        order = [i1] + others[:min(k, len(others), V - 1)]
        first_az[b] = az[i1]
        for s, j in enumerate(order):
            im = recs[b]["imgs"][j].astype(np.float32) / 255.0
            if im.shape[0] != S:
                im = F.interpolate(torch.from_numpy(im).permute(2, 0, 1)[None], size=(S, S), mode="bilinear", antialias=True)[0].permute(1, 2, 0).numpy()
            imgs[b, s] = im.transpose(2, 0, 1) * 2 - 1
            rel = ((az[j] - az[i1] + 180.0) % 360.0) - 180.0
            if abs(rel) > 179.9:
                rel = 180.0
            vp[b, s] = (np.radians(rel), np.radians(el[j]))
            vm[b, s] = True
        ip = _take(r["pts"], vc.n_points, rng)
        pts.append(r["pts"][ip].astype(np.float32))
        hc.append(float(r["has_colour"]))
    pts = torch.from_numpy(np.stack(pts)).to(device)
    R = rot_y_batch(torch.from_numpy(first_az).to(device))
    pts = torch.cat([_rot_pts(pts[..., :3], R), _rot_pts(pts[..., 3:6], R), pts[..., 6:]], -1)
    return pts, torch.tensor(hc, device=device), torch.from_numpy(imgs).to(device), torch.from_numpy(vp).to(device), torch.from_numpy(vm).to(device)


class _Log:
    def __init__(self, path: Optional[str]):
        self.f = open(path, "a") if path else None

    def __call__(self, d: Dict):
        if self.f:
            self.f.write(json.dumps(d) + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def train_vae(model: AuraNative, ds: ShardDataset, steps: int = 2000, batch: int = 16, lr: float = 3e-4, device="auto",
              seed: int = 0, yaw: float = 0.0, log_path: Optional[str] = None, ckpt: Optional[str] = None, say=print,
              save_every: int = 1000, ema_decay: float = 0.999, start_step: int = 0, stop_after: Optional[float] = None) -> Dict:
    """`start_step` > 0 resumes (the model already holds the weights; the optimiser restarts with a short ramp).
    `stop_after` (seconds) ends the call early and saves - run it repeatedly inside tools that have a time limit.
    Returns the last losses plus {"next_step": ..., "done": bool}."""
    dev = pick_device(device) if isinstance(device, str) else device
    model.to(dev)
    vae = model.vae.train()
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
    ema = EMA(vae, ema_decay)
    rng = np.random.default_rng(seed + 7919 * start_step)
    torch.manual_seed(seed + start_step)
    log = _Log(log_path)
    t0, last = time.time(), {}
    step, broke = start_step, False
    for step in range(start_step, steps):
        ramp = min(1.0, (step - start_step + 1) / 20.0) if start_step else 1.0
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr) * ramp
        if stop_after is not None and time.time() - t0 > stop_after:
            broke = True
            break
        recs = [ds[int(i)] for i in rng.integers(0, len(ds), size=batch)]
        pts, q, sdf, rgb, hc = vae_batch(recs, model.cfg.vae, rng, dev, yaw)
        with _autocast(dev):
            out = vae.loss(pts, q, sdf, rgb, hc)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        ema.update(vae)
        last = {k: float(v.detach()) for k, v in out.items()}
        if step % 50 == 0 or step == steps - 1:
            rec = {"stage": "vae", "step": step, **{k: round(v, 5) for k, v in last.items()}, "sec": round(time.time() - t0, 1)}
            log(rec)
            if say and (step % 200 == 0 or step == steps - 1):
                say(f"  vae {step:5d}/{steps}  total {last['total']:.4f}  sdf {last['sdf']:.4f}  occ {last['occ']:.4f}  rgb {last.get('rgb', 0):.4f}  ({time.time() - t0:.0f}s)")
        if ckpt and save_every and step and step % save_every == 0:
            _save_ema(model, ema, ckpt, "vae", step)
    done = not broke
    next_step = steps if done else step
    ema.copy_to(vae)
    vae.eval()
    log.close()
    if ckpt:
        model.save(ckpt, {"vae_steps": next_step})
    return {**last, "next_step": next_step, "done": bool(done)}


def _save_ema(model, ema, ckpt, which, step):
    mod = model.vae if which == "vae" else model.flow
    keep = {k: v.clone() for k, v in mod.state_dict().items()}
    ema.copy_to(mod)
    model.save(ckpt, {f"{which}_steps_so_far": step})
    mod.load_state_dict(keep)


@torch.no_grad()
def fit_latent_stats(model: AuraNative, ds: ShardDataset, n: int = 256, seed: int = 0, device=None):
    """Per-channel mean / std of the posterior means, in each asset's first-view frame (the generator's target space)."""
    dev = model.device
    model.eval()
    rng = np.random.default_rng(seed)
    mus = []
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    for k in range(0, len(idx), 16):
        recs = [ds[int(i)] for i in idx[k:k + 16]]
        pts, hc, *_ = flow_batch(recs, model.cfg, rng, dev, front_prob=1.0)
        mu, _ = model.vae.encode(pts, hc)
        mus.append(mu.float().cpu())
    mu = torch.cat(mus, 0)                                                # (n, C, G, G, G)
    model.mean = mu.mean((0, 2, 3, 4)).to(dev)
    model.std = mu.std((0, 2, 3, 4)).clamp(min=1e-3).to(dev)
    return model.mean.cpu().tolist(), model.std.cpu().tolist()


def train_flow(model: AuraNative, ds: ShardDataset, steps: int = 3000, batch: int = 16, lr: float = 3e-4, device="auto",
               seed: int = 0, front_prob: float = 0.8, log_path: Optional[str] = None, ckpt: Optional[str] = None, say=print,
               save_every: int = 1000, ema_decay: float = 0.999, sample_weights: Optional[np.ndarray] = None,
               init_from_ema: bool = False, base_ds: Optional[ShardDataset] = None, base_frac: float = 0.0,
               start_step: int = 0, stop_after: Optional[float] = None) -> Dict:
    """Train the generator.  `sample_weights` (len(ds)) draws assets proportionally and `base_ds`/`base_frac` mixes in a
    replay set of trusted data - the two levers the self-improvement loop uses (see evolve.py)."""
    dev = pick_device(device) if isinstance(device, str) else device
    model.to(dev)
    for p in model.vae.parameters():
        p.requires_grad_(False)
    model.vae.eval()
    flow = model.flow.train()
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
    ema = EMA(flow, ema_decay)
    if init_from_ema:
        ema.shadow = {k: v.detach().clone() for k, v in flow.state_dict().items()}
    rng = np.random.default_rng(seed + 7919 * start_step)
    torch.manual_seed(seed + start_step)
    log = _Log(log_path)
    p = None if sample_weights is None else np.asarray(sample_weights, np.float64) / np.sum(sample_weights)
    t0, last = time.time(), {}
    step, broke = start_step, False
    for step in range(start_step, steps):
        ramp = min(1.0, (step - start_step + 1) / 20.0) if start_step else 1.0
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr, warmup=min(100, steps // 10 + 1)) * ramp
        if stop_after is not None and time.time() - t0 > stop_after:
            broke = True
            break
        recs = []
        for _ in range(batch):
            if base_ds is not None and rng.random() < base_frac:
                recs.append(base_ds[int(rng.integers(len(base_ds)))])
            else:
                recs.append(ds[int(rng.choice(len(ds), p=p))])
        pts, hc, imgs, vp, vm = flow_batch(recs, model.cfg, rng, dev, front_prob)
        with torch.no_grad():
            mu, logvar = model.vae.encode(pts, hc)
            z = model.vae.sample(mu, logvar)
            z1 = (z - model.mean.view(1, -1, 1, 1, 1)) / model.std.view(1, -1, 1, 1, 1)
        with _autocast(dev):
            loss = flow.loss(z1, imgs, vp, vm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        ema.update(flow)
        last = {"loss": float(loss.detach())}
        if step % 50 == 0 or step == steps - 1:
            log({"stage": "flow", "step": step, "loss": round(last["loss"], 5), "sec": round(time.time() - t0, 1)})
            if say and (step % 250 == 0 or step == steps - 1):
                say(f"  flow {step:5d}/{steps}  loss {last['loss']:.4f}  ({time.time() - t0:.0f}s)")
        if ckpt and save_every and step and step % save_every == 0:
            _save_ema(model, ema, ckpt, "flow", step)
    done = not broke
    next_step = steps if done else step
    ema.copy_to(flow)
    flow.eval()
    log.close()
    if ckpt:
        model.save(ckpt, {"flow_steps": next_step})
    return {**last, "next_step": next_step, "done": bool(done)}
