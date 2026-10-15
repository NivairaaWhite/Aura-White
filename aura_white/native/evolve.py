"""Self-improvement with a leash: generate -> verify -> keep the winners -> fine-tune -> promote only on evidence.

What "the model studies its own output" can and cannot mean (read this before turning the knob up):
  * Training a generator on its own raw samples degrades it ("model collapse").  Training on samples that an EXTERNAL
    verifier accepted can help - it sharpens the model toward what the verifier rewards - but analysis of verified
    self-training shows the gains plateau at what the verifier itself can recognise, and early gains can reverse if the
    verifier is biased.  So the verifier here is made of ground-truth-free *physical* checks (gates.py: real depth, a back
    that is not a mirror, closed mesh, agreement with the picture) and it is never the only judge.
  * Each round therefore keeps four guards:   (1) the verifier picks the winners (never the model itself);
      (2) a fixed share of every training batch is ANCHOR data (real / teacher assets) so the model cannot drift;
      (3) the challenger is scored on a FROZEN held-out set with ground truth, against the champion, with the same seeds;
      (4) it is promoted only if the oracle numbers improve AND diversity has not collapsed - otherwise it is discarded.
  * The loop needs pictures, not 3-D assets: any pile of unlabeled concept art can be the prompt pool.  That is where it
    adds something a plain "train longer on the labelled data" run cannot.

Everything is logged to `evolution.jsonl`; a rolled-back round is a result, not an error.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .bench import BenchItem, evaluate
from .data import ShardDataset
from .gates import GateConfig, TargetView, verify
from .model import AuraNative, ViewIn
from .train import EMA, cosine_lr, flow_batch, pick_device


@dataclass
class EvolveConfig:
    rounds: int = 2
    prompts: int = 120                # pictures drawn from the pool each round
    k: int = 4                        # candidates per picture
    selection: str = "verifier"       # verifier | random   (random = the "no verifier" control, same count of winners)
    score_floor: float = 0.75         # a candidate must beat this verifier score to be eligible
    anchor_frac: float = 0.5          # share of each training batch that is anchor (trusted) data
    reward_weight: bool = True        # weight winners by how far above the floor they scored
    finetune_steps: int = 600
    batch: int = 16
    lr: float = 2e-4
    ema: float = 0.99
    steps: int = 16                   # sampler steps while generating
    cfg_scale: float = 3.0
    res: int = 40                     # decode resolution while generating (cheap; the verifier does not need more)
    margin_se: float = 0.0            # promote if oracle chamfer improves by more than margin_se standard errors
    diversity_floor: float = 0.5      # ...and diversity stays above this share of the champion's
    bench_seeds: int = 3
    seed: int = 0
    always_promote: bool = False      # experiments only: take every challenger to see the raw dynamics (the rule is still logged)

    def as_dict(self):
        return asdict(self)


@dataclass
class Cand:
    prompt: int
    views: List[ViewIn]
    z: np.ndarray                     # (C,G,G,G) normalised latent
    score: float
    passed: bool
    seed: int
    tens: Optional[tuple] = None      # (imgs (V,4,S,S), vparams (V,2), vmask (V,)) cpu tensors


def prompt_views(ds: ShardDataset, i: int, az_el=((0.0, 0.0),)) -> List[ViewIn]:
    rec = ds[i]
    out = []
    for az, el in az_el:
        j = int(np.argmin(np.abs(rec["view_az"] - az) + np.abs(rec["view_el"] - el)))
        out.append(ViewIn(rec["imgs"][j].astype(np.float32) / 255.0, float(az), float(el)))
    return out


@torch.no_grad()
def generate_candidates(model: AuraNative, pool: ShardDataset, idxs: Sequence[int], cfg: EvolveConfig,
                        gate_cfg: Optional[GateConfig] = None, say: Optional[Callable] = None, seed0: int = 0) -> List[Cand]:
    gate_cfg = gate_cfg or GateConfig(size=64)
    out: List[Cand] = []
    t0 = time.time()
    for n, pi in enumerate(idxs):
        views = prompt_views(pool, int(pi))
        imgs, vp, vm = model.views_to_tensors(views)
        Z = model.sample_z(views, n=cfg.k, steps=cfg.steps, cfg_scale=cfg.cfg_scale, seed=seed0 * 100003 + int(pi))
        tv = [TargetView(views[0].image, 0.0, 0.0, "front")]
        for j in range(cfg.k):
            v, f, c = model.decode_mesh(Z[j:j + 1], res=cfg.res)
            if len(f) < 8:
                out.append(Cand(int(pi), views, Z[j].cpu().numpy(), 0.0, False, j, (imgs[0].cpu(), vp[0].cpu(), vm[0].cpu())))
                continue
            ver = verify(v, f, c, tv, expect_solid=True, cfg=gate_cfg)
            out.append(Cand(int(pi), views, Z[j].cpu().numpy(), ver.score, ver.passed, j, (imgs[0].cpu(), vp[0].cpu(), vm[0].cpu())))
        if say and (n + 1) % 20 == 0:
            say(f"    generated {n + 1}/{len(idxs)} pictures ({time.time() - t0:.0f}s)")
    return out


def select_winners(cands: Sequence[Cand], cfg: EvolveConfig, rng: np.random.Generator, n_match: Optional[int] = None) -> List[Cand]:
    by: Dict[int, List[Cand]] = {}
    for c in cands:
        by.setdefault(c.prompt, []).append(c)
    if cfg.selection == "verifier":
        win = []
        for lst in by.values():
            b = max(lst, key=lambda c: c.score)
            if b.score >= cfg.score_floor:
                win.append(b)
        return win
    if cfg.selection == "random":                                    # control: no verifier, same number of winners
        keys = list(by)
        n = n_match if n_match is not None else len(keys)
        rng.shuffle(keys)
        return [by[k][int(rng.integers(len(by[k])))] for k in keys[:n]]
    raise ValueError(cfg.selection)


def finetune(model: AuraNative, anchor: ShardDataset, winners: Sequence[Cand], cfg: EvolveConfig, device=None, say=None) -> Dict:
    """Short flow fine-tune on a mix of anchor data and the winners (the model passed in is updated in place)."""
    dev = model.device if device is None else device
    model.to(dev)
    for p in model.vae.parameters():
        p.requires_grad_(False)
    model.vae.eval()
    flow = model.flow.train()
    for p in flow.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(flow.parameters(), lr=cfg.lr, weight_decay=0.01, betas=(0.9, 0.99))
    ema = EMA(flow, cfg.ema)
    rng = np.random.default_rng(cfg.seed + 12345)
    torch.manual_seed(cfg.seed + 12345)
    have = len(winners) > 0 and cfg.anchor_frac < 1.0
    if have:
        W_imgs = torch.stack([c.tens[0] for c in winners])
        W_vp = torch.stack([c.tens[1] for c in winners])
        W_vm = torch.stack([c.tens[2] for c in winners])
        W_z = torch.from_numpy(np.stack([c.z for c in winners])).float()
        sc = np.array([c.score for c in winners])
        W_w = (torch.from_numpy(np.clip((sc - cfg.score_floor) / max(1e-6, 1.0 - cfg.score_floor), 0.0, 1.0) * 1.5 + 0.5).float()
               if cfg.reward_weight else torch.ones(len(winners)))
    t0, loss_v = time.time(), 0.0
    for step in range(cfg.finetune_steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg.finetune_steps, cfg.lr, warmup=20)
        n_anchor = int(sum(rng.random() < cfg.anchor_frac for _ in range(cfg.batch))) if have else cfg.batch
        parts = []
        if n_anchor:
            recs = [anchor[int(i)] for i in rng.integers(0, len(anchor), size=n_anchor)]
            pts, hc, imgs, vp, vm = flow_batch(recs, model.cfg, rng, dev, front_prob=0.9)
            with torch.no_grad():
                mu, lv = model.vae.encode(pts, hc)
                z = model.vae.sample(mu, lv)
                z1 = (z - model.mean.view(1, -1, 1, 1, 1)) / model.std.view(1, -1, 1, 1, 1)
            parts.append((z1, imgs, vp, vm, torch.ones(n_anchor, device=dev)))
        if have and cfg.batch - n_anchor > 0:
            ix = torch.from_numpy(rng.integers(0, len(winners), size=cfg.batch - n_anchor))
            parts.append((W_z[ix].to(dev), W_imgs[ix].to(dev), W_vp[ix].to(dev), W_vm[ix].to(dev), W_w[ix].to(dev)))
        z1, imgs, vp, vm, w = (torch.cat([p[i] for p in parts], 0) for i in range(5))
        loss = flow.loss(z1, imgs, vp, vm, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        ema.update(flow)
        loss_v = float(loss.detach())
    ema.copy_to(flow)
    flow.eval()
    return {"final_loss": loss_v, "steps": cfg.finetune_steps, "sec": round(time.time() - t0, 1), "winners": len(winners)}


def promote(champ: Dict, chal: Dict, cfg: EvolveConfig):
    """Decision rule on the frozen held-out set (oracle numbers, ground truth).  Returns (bool, reason)."""
    a, b = champ["summary"], chal["summary"]
    d_ch = a["chamfer"] - b["chamfer"]                                   # positive = challenger better
    se = float(np.hypot(a["chamfer_se"], b["chamfer_se"]))
    if d_ch <= cfg.margin_se * se:
        return False, f"chamfer did not improve beyond {cfg.margin_se:g} SE ({d_ch:+.4f}, SE {se:.4f})"
    if b["f06"] < a["f06"] - 0.01:
        return False, f"F-score dropped ({b['f06']:.3f} < {a['f06']:.3f})"
    if np.isfinite(a["diversity"]) and np.isfinite(b["diversity"]) and b["diversity"] < cfg.diversity_floor * a["diversity"]:
        return False, f"diversity collapsed ({b['diversity']:.3f} < {cfg.diversity_floor:g} x {a['diversity']:.3f})"
    return True, f"chamfer {a['chamfer']:.4f} -> {b['chamfer']:.4f}"


def evolve(champion: AuraNative, anchor: ShardDataset, pool: ShardDataset, bench_items: Sequence[BenchItem], cfg: EvolveConfig,
           out_dir: str, shared_round1: Optional[List[Cand]] = None, say: Callable = print, round_offset: int = 0,
           champion_result: Optional[Dict] = None) -> Dict:
    """Run `cfg.rounds` rounds.  Returns {"history": [...], "champion": AuraNative}.  `shared_round1`: pre-generated round-1
    candidates (arms that start from the same checkpoint can share them)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    seeds = tuple(range(cfg.bench_seeds))
    log = open(out / "evolution.jsonl", "a")
    base = champion_result or evaluate(champion, bench_items, seeds=seeds, steps=cfg.steps, cfg_scale=cfg.cfg_scale)
    history = [{"round": round_offset, "summary": base["summary"]}]
    log.write(json.dumps({"round": round_offset, "summary": base["summary"], "config": cfg.as_dict()}, default=float) + "\n")
    log.flush()
    champ_res = base
    perm = np.random.default_rng(cfg.seed).permutation(len(pool))
    for r0 in range(1, cfg.rounds + 1):
        r = round_offset + r0
        t0 = time.time()
        idxs = perm[(r - 1) * cfg.prompts: r * cfg.prompts]
        if len(idxs) < cfg.prompts:
            idxs = rng.choice(len(pool), size=cfg.prompts, replace=False)
        winners: List[Cand] = []
        gen_stats = {}
        if cfg.anchor_frac < 1.0:
            cands = shared_round1 if (r == 1 and shared_round1 is not None) else generate_candidates(champion, pool, idxs, cfg, say=say, seed0=r)
            sc = np.array([c.score for c in cands]) if cands else np.zeros(1)
            nv = len(select_winners(cands, EvolveConfig(**{**cfg.as_dict(), "selection": "verifier"}), rng))
            winners = select_winners(cands, cfg, rng, n_match=nv)
            gen_stats = {"candidates": len(cands), "mean_score": float(sc.mean()), "pass_rate": float(np.mean([c.passed for c in cands])),
                         "winners": len(winners), "winner_mean_score": float(np.mean([c.score for c in winners])) if winners else float("nan")}
        challenger = champion.clone()
        ft = finetune(challenger, anchor, winners, cfg)
        chal_res = evaluate(challenger, bench_items, seeds=seeds, steps=cfg.steps, cfg_scale=cfg.cfg_scale)
        ok_rule, why = promote(champ_res, chal_res, cfg)
        ok = ok_rule or cfg.always_promote
        rec = {"round": r, "generation": gen_stats, "finetune": ft, "challenger": chal_res["summary"], "champion_before": champ_res["summary"],
               "promoted": bool(ok), "rule_would_promote": bool(ok_rule), "reason": why, "sec": round(time.time() - t0, 1)}
        history.append(rec)
        log.write(json.dumps(rec, default=float) + "\n")
        log.flush()
        say(f"  round {r}: {'PROMOTED' if ok else 'rolled back'} - {why}   (chamfer {champ_res['summary']['chamfer']:.4f} -> {chal_res['summary']['chamfer']:.4f}, "
            f"score {champ_res['summary']['score']:.3f} -> {chal_res['summary']['score']:.3f}, pass {champ_res['summary']['pass_rate']:.2f} -> {chal_res['summary']['pass_rate']:.2f})")
        if ok:
            champion, champ_res = challenger, chal_res
    log.close()
    return {"history": history, "champion": champion, "champion_result": champ_res}
