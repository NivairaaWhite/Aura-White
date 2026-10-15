"""Training loop for Aura Shape.  Data = procedural shapes (built in, unlimited) or a folder of your own meshes."""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Iterator, Optional

import numpy as np
import torch

from . import sdf as S
from .model import FlowMatching, ShapeDiT


def procedural_batches(res: int, batch: int, seed: int = 0) -> Iterator[tuple]:
    rng = np.random.default_rng(seed)
    while True:
        xs, cs, ms = [], [], []
        for _ in range(batch):
            sd = S.random_shape(res, rng)
            # random view axis / side so the model does not learn one fixed camera
            axis, pos = int(rng.integers(3)), bool(rng.integers(2))
            obs = S.observed_mask(sd, thickness=int(rng.integers(1, 4)), axis=axis, from_positive=pos)
            t = S.to_tsdf(sd, res)
            xs.append(t)
            cs.append(np.where(obs, t, 0.0))
            ms.append(obs.astype(np.float32))
        yield tuple(torch.from_numpy(np.stack(a).astype(np.float32))[:, None] for a in (xs, cs, ms))


def mesh_folder_batches(folder: str, res: int, batch: int, seed: int = 0) -> Iterator[tuple]:
    """Batches from *.glb / *.obj meshes in `folder` (each fitted to [-0.8,0.8]^3, randomly rotated)."""
    from ..backends.glb import read_glb

    files = [os.path.join(dp, f) for dp, _, fs in os.walk(folder) for f in fs if f.lower().endswith((".glb", ".obj"))]
    if not files:
        raise FileNotFoundError(f"no .glb / .obj meshes under {folder}")
    rng = np.random.default_rng(seed)
    while True:
        xs, cs, ms = [], [], []
        while len(xs) < batch:
            try:
                path = files[int(rng.integers(len(files)))]
                if path.lower().endswith(".glb"):
                    v, f = read_glb(path)
                else:
                    v, f = _read_obj(path)
                v = v.astype(np.float64) - 0.5 * (v.max(0) + v.min(0))
                v *= 0.8 / max(float(np.abs(v).max()), 1e-9)
                v = v @ S._rot(rng).T
                sd = S.mesh_to_sdf(v, f, res, seed=int(rng.integers(1 << 30)))
            except Exception:
                continue
            obs = S.observed_mask(sd, thickness=int(rng.integers(1, 4)), axis=int(rng.integers(3)),
                                  from_positive=bool(rng.integers(2)))
            t = S.to_tsdf(sd, res)
            xs.append(t)
            cs.append(np.where(obs, t, 0.0))
            ms.append(obs.astype(np.float32))
        yield tuple(torch.from_numpy(np.stack(a).astype(np.float32))[:, None] for a in (xs, cs, ms))


def _read_obj(path: str):
    v, f = [], []
    for line in open(path, errors="ignore"):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) for p in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(v) + i for i in idx]
            for k in range(1, len(idx) - 1):
                f.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


@torch.no_grad()
def evaluate(model: ShapeDiT, res: int, n: int = 8, steps: int = 16, seed: int = 999, device="cpu") -> dict:
    """IoU of the generated inside vs the truth, and the same for 'no completion' (observed voxels only)."""
    flow = FlowMatching()
    rng = np.random.default_rng(seed)
    ious, base = [], []
    for i in range(n):
        sd = S.random_shape(res, rng)
        obs = S.observed_mask(sd, thickness=2, axis=0, from_positive=True)
        t = S.to_tsdf(sd, res)
        cond = torch.from_numpy(np.where(obs, t, 0.0).astype(np.float32))[None, None].to(device)
        m = torch.from_numpy(obs.astype(np.float32))[None, None].to(device)
        out = flow.sample(model, cond, m, steps=steps, seed=i)[0, 0].cpu().numpy()
        gt = t < 0
        pred = out < 0
        ious.append((pred & gt).sum() / max((pred | gt).sum(), 1))
        no_fill = np.where(obs, t, 1.0) < 0
        base.append((no_fill & gt).sum() / max((no_fill | gt).sum(), 1))
    return {"iou": float(np.mean(ious)), "iou_no_completion": float(np.mean(base))}


def train(steps: int = 2000, res: int = 32, patch: int = 4, d_model: int = 256, layers: int = 8, heads: int = 8,
          batch: int = 16, lr: float = 3e-4, data: str = "procedural", out: str = "aura_shape.pt", device: str = "auto",
          ema_decay: float = 0.999, log_every: int = 50, save_every: int = 1000, seed: int = 0, resume: Optional[str] = None,
          progress=print) -> ShapeDiT:
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    torch.manual_seed(seed)
    model = ShapeDiT(res, patch, d_model, layers, heads).to(dev)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    step0 = 0
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck.get("ema") or ck["model"])
        step0 = int(ck.get("step", 0))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    flow = FlowMatching()
    stream = procedural_batches(res, batch, seed) if data == "procedural" else mesh_folder_batches(data, res, batch, seed)
    amp = dev.type == "cuda"
    t0, run = time.time(), None
    for step in range(step0, steps):
        x1, cond, mask = (a.to(dev) for a in next(stream))
        lr_now = lr * min(1.0, (step + 1) / 100) * (0.5 * (1 + np.cos(np.pi * step / max(steps, 1))) * 0.9 + 0.1)
        for g_ in opt.param_groups:
            g_["lr"] = lr_now
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            loss = flow.loss(model, x1, cond, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(ema_decay, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        run = float(loss.detach()) if run is None else 0.98 * run + 0.02 * float(loss.detach())
        if step % log_every == 0:
            progress(f"step {step:6d}  loss {run:.4f}  lr {lr_now:.2e}  {time.time() - t0:.0f}s")
        if (step + 1) % save_every == 0 or step + 1 == steps:
            torch.save({"config": model.config(), "model": model.state_dict(), "ema": ema.state_dict(), "step": step + 1}, out)
    return ema


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aura-white shape-train")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--data", default="procedural", help="'procedural' or a folder of .glb/.obj meshes (Objaverse, ...)")
    ap.add_argument("--out", default="aura_shape.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    train(a.steps, a.res, a.patch, a.d_model, a.layers, a.heads, a.batch, a.lr, a.data, a.out, a.device, resume=a.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
