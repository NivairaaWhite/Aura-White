"""The generator: a rectified-flow transformer over the ShapeVAE's latent tokens, conditioned on one or several pictures.

  pictures -> PatchViT (own small ViT, trained from scratch; a frozen external encoder can be plugged in) -> image tokens
  noise z_0 ~ N(0, I)  --flow ODE (Euler / Heun, classifier-free guidance)-->  z_1  --ShapeVAE decoder-->  SDF + colour field

Because the generator is a *flow* (it samples), a hidden back side is drawn from what is plausible rather than averaged
into a blur or mirrored from the front - and a picture of the back, when you have one, simply becomes more tokens to
attend to.  Training objective:  x_t = (1-t) z_0 + t z_1,  v* = z_1 - z_0   (t = 1 is data), logit-normal t.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FlowConfig, VAEConfig
from .vae import MHA, MLP, SelfBlock


def sincos_3d(d_model: int, g: int) -> torch.Tensor:
    """(g^3, d_model) fixed 3-D sine-cosine position embedding (raster order x, y, z)."""
    dim = (d_model // 3) // 2 * 2
    om = 1.0 / (10000 ** (torch.arange(dim // 2, dtype=torch.float32) / max(dim // 2, 1)))
    e = torch.arange(g).float()[:, None] * om[None]
    e = torch.cat([e.sin(), e.cos()], 1)                                  # (g, dim)
    x = e[:, None, None].expand(g, g, g, dim)
    y = e[None, :, None].expand(g, g, g, dim)
    z = e[None, None, :].expand(g, g, g, dim)
    pe = torch.cat([x, y, z], -1).reshape(g ** 3, 3 * dim)
    return F.pad(pe, (0, d_model - pe.shape[1])) if pe.shape[1] < d_model else pe


def patchify(z: torch.Tensor, p: int) -> torch.Tensor:
    """latent grid (B,C,G,G,G) -> tokens (B,(G/p)^3,p^3*C)."""
    B, C, G = z.shape[0], z.shape[1], z.shape[2]
    g = G // p
    z = z.view(B, C, g, p, g, p, g, p).permute(0, 2, 4, 6, 3, 5, 7, 1)
    return z.reshape(B, g ** 3, p ** 3 * C)


def unpatchify(t: torch.Tensor, p: int, C: int) -> torch.Tensor:
    B, M, _ = t.shape
    g = round(M ** (1 / 3))
    t = t.view(B, g, g, g, p, p, p, C).permute(0, 7, 1, 4, 2, 5, 3, 6)
    return t.reshape(B, C, g * p, g * p, g * p)


class TimeEmbed(nn.Module):
    def __init__(self, d: int, freqs: int = 64):
        super().__init__()
        self.register_buffer("f", torch.exp(-math.log(10000) * torch.arange(freqs) / freqs), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * freqs, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t.float()[:, None] * 1000.0 * self.f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))             # float32 in; autocast handles half precision


class PatchViT(nn.Module):
    """Own image encoder: patchify -> + position + view embedding (azimuth / elevation) -> self-attention blocks."""

    def __init__(self, cfg: FlowConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        g = cfg.image_size // cfg.patch
        self.T = g * g
        self.patch = nn.Conv2d(4, d, cfg.patch, stride=cfg.patch)
        self.pos = nn.Parameter(torch.randn(1, self.T, d) * 0.02)
        self.view = nn.Sequential(nn.Linear(4, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([SelfBlock(d, cfg.heads) for _ in range(cfg.img_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, imgs: torch.Tensor, vparams: torch.Tensor) -> torch.Tensor:
        """imgs (B,V,4,S,S) in [-1,1]; vparams (B,V,2) azimuth, elevation in radians -> tokens (B,V*T,d)."""
        B, V = imgs.shape[:2]
        x = self.patch(imgs.flatten(0, 1)).flatten(2).transpose(1, 2) + self.pos                # (B*V,T,d)
        vp = vparams.flatten(0, 1)
        feat = torch.stack([vp[:, 0].sin(), vp[:, 0].cos(), vp[:, 1].sin(), vp[:, 1].cos()], -1)
        x = x + self.view(feat.to(x.dtype)).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).view(B, V * self.T, -1)


class FlowBlock(nn.Module):
    def __init__(self, d: int, heads: int, ratio: float):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.nc = nn.LayerNorm(d)
        self.attn, self.xattn, self.mlp = MHA(d, heads), MHA(d, heads), MLP(d, ratio)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, ctx, mask):
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.ada(c)[:, None].chunk(9, -1)
        x = x + g1 * self.attn(self.n1(x) * (1 + c1) + s1)
        x = x + g2 * self.xattn(self.n2(x) * (1 + c2) + s2, self.nc(ctx), mask)
        return x + g3 * self.mlp(self.n3(x) * (1 + c3) + s3)


class LatentFlow(nn.Module):
    def __init__(self, vcfg: VAEConfig, cfg: FlowConfig):
        super().__init__()
        self.cfg, self.vcfg = cfg, vcfg
        d, C, p = cfg.d_model, vcfg.latent_dim, cfg.patch3
        if vcfg.grid % p:
            raise ValueError("latent grid must be divisible by the flow patch size")
        self.p, self.C, self.g = p, C, vcfg.grid // p
        self.inp = nn.Linear(p ** 3 * C, d)
        self.register_buffer("pos", sincos_3d(d, self.g)[None], persistent=False)
        self.time = TimeEmbed(d)
        self.w_embed = TimeEmbed(d)                                    # guidance-conditioned (distilled) models only
        nn.init.zeros_(self.w_embed.mlp[2].weight)
        nn.init.zeros_(self.w_embed.mlp[2].bias)
        self.img = PatchViT(cfg)
        self.null = nn.Parameter(torch.randn(1, 1, d) * 0.02)          # always-valid token: what "no picture" attends to
        self.pool = nn.Linear(d, d)
        self.blocks = nn.ModuleList([FlowBlock(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.layers)])
        self.n_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, p ** 3 * C)
        for m in (self.ada_out[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    # -- conditioning
    def condition(self, imgs: torch.Tensor, vparams: torch.Tensor, vmask: torch.Tensor,
                  drop: Optional[torch.Tensor] = None):
        """-> ctx (B,1+V*T,d), mask (B,1,1,1+V*T) bool (True = attend), pooled (B,d).  `drop` (B,) bool: unconditional."""
        B, V = imgs.shape[:2]
        tok = self.img(imgs, vparams)
        T = self.img.T
        m = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=imgs.device), vmask.repeat_interleave(T, 1)], 1)
        if drop is not None:
            keep = torch.zeros_like(m)
            keep[:, 0] = True
            m = torch.where(drop.view(B, 1), keep, m)
        ctx = torch.cat([self.null.expand(B, -1, -1).to(tok.dtype), tok], 1)
        mm = m[:, 1:].unsqueeze(-1).to(tok.dtype)
        pooled = self.pool((tok * mm).sum(1) / mm.sum(1).clamp(min=1.0))
        return ctx, m[:, None, None, :], pooled

    def forward(self, z: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor, pooled: torch.Tensor,
                w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z: latent GRID (B,C,G,G,G) -> velocity, same shape."""
        c = self.time(t) + pooled
        if w is not None:
            c = c + self.w_embed(w)
        x = self.inp(patchify(z, self.p)) + self.pos.to(z.dtype)
        for blk in self.blocks:
            x = blk(x, c, ctx, mask)
        s, sc = self.ada_out(c)[:, None].chunk(2, -1)
        return unpatchify(self.out(self.n_out(x) * (1 + sc) + s), self.p, self.C)

    # -- training
    def loss(self, z1: torch.Tensor, imgs, vparams, vmask, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = z1.shape[0]
        drop = torch.rand(B, device=z1.device) < self.cfg.cond_drop
        ctx, mask, pooled = self.condition(imgs, vparams, vmask, drop)
        t = torch.sigmoid(torch.randn(B, device=z1.device))
        z0 = torch.randn_like(z1)
        tt = t.view(B, 1, 1, 1, 1)
        v = self(((1 - tt) * z0 + tt * z1), t, ctx, mask, pooled)
        per = ((v.float() - (z1 - z0).float()) ** 2).mean((1, 2, 3, 4))
        if weights is not None:
            per = per * weights
        return per.mean()


@torch.no_grad()
def sample_latents(flow: LatentFlow, imgs, vparams, vmask, n: int = 1, steps: int = 24, cfg_scale: float = 3.0,
                   seed: int = 0, heun: bool = True, w_embed: bool = False) -> torch.Tensor:
    """Draw `n` latents for ONE conditioning set (imgs (1,V,4,S,S)).  Returns (n,C,G,G,G) in normalised latent units.
    `w_embed=True`: the model has guidance distilled in (one forward per step, no unconditional pass)."""
    dev = imgs.device
    G, C = flow.vcfg.grid, flow.vcfg.latent_dim
    g = torch.Generator(device=dev).manual_seed(int(seed))
    x = torch.randn(n, C, G, G, G, generator=g, device=dev)
    rep = lambda a: a.expand(n, *a.shape[1:])
    use_cfg = cfg_scale != 1.0 and not w_embed
    ctx, mask, pooled = flow.condition(imgs, vparams, vmask)
    ctx, mask, pooled = rep(ctx), rep(mask), rep(pooled)
    if use_cfg:
        ctx_u, mask_u, pooled_u = flow.condition(imgs, vparams, vmask, torch.ones(1, dtype=torch.bool, device=dev))
        ctx_u, mask_u, pooled_u = rep(ctx_u), rep(mask_u), rep(pooled_u)
    wv = torch.full((n,), float(cfg_scale), device=dev) if w_embed else None

    def vel(xx, t):
        tb = t.expand(n)
        vc = flow(xx, tb, ctx, mask, pooled, wv)
        if not use_cfg:
            return vc
        vu = flow(xx, tb, ctx_u, mask_u, pooled_u)
        return vu + cfg_scale * (vc - vu)

    ts = torch.linspace(0, 1, steps + 1, device=dev)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        v = vel(x, t0)
        xn = x + (t1 - t0) * v
        if heun and i < steps - 1:
            xn = x + (t1 - t0) * 0.5 * (v + vel(xn, t1))
        x = xn
    return x
