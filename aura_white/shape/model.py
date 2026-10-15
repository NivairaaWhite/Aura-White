"""Conditional flow-matching Diffusion Transformer over a dense truncated-SDF volume.

  input channels : x_t (the noisy SDF being generated), observed SDF (0 where unseen), observed mask
  tokens         : non-overlapping p^3 patches -> (R/p)^3 tokens, fixed 3-D sin-cos positions
  conditioning   : adaLN-Zero on the flow time t (Peebles & Xie), zero-initialised residual gates
  objective      : rectified flow  x_t = (1-t) x0 + t x1,  v* = x1 - x0   (x0 ~ N(0,I), x1 = clean TSDF)
  sampling       : Euler / Heun ODE from t=0 to 1 with hard data consistency on the observed voxels
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = 1.0 / (10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)))
    out = pos.float()[:, None] * omega[None]
    return torch.cat([out.sin(), out.cos()], 1)


def sincos_3d(d_model: int, g: int) -> torch.Tensor:
    """(g^3, d_model) fixed 3-D positional embedding."""
    dim = (d_model // 3) // 2 * 2
    ax = torch.arange(g)
    e = _sincos_1d(dim, ax)                                    # (g, dim)
    x = e[:, None, None].expand(g, g, g, dim)
    y = e[None, :, None].expand(g, g, g, dim)
    z = e[None, None, :].expand(g, g, g, dim)
    pe = torch.cat([x, y, z], -1).reshape(g ** 3, 3 * dim)
    if pe.shape[1] < d_model:
        pe = F.pad(pe, (0, d_model - pe.shape[1]))
    return pe


class TimeEmbed(nn.Module):
    def __init__(self, d: int, freqs: int = 128):
        super().__init__()
        self.register_buffer("f", torch.exp(-math.log(10000) * torch.arange(freqs) / freqs), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * freqs, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t[:, None] * 1000.0 * self.f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp: float = 4.0):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp)), nn.GELU(approximate="tanh"), nn.Linear(int(d * mlp), d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.ada(c)[:, None].chunk(6, -1)
        B, N, D = x.shape
        h = self.n1(x) * (1 + sc1) + s1
        q, k, v = self.qkv(h).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + g1 * self.proj(a)
        x = x + g2 * self.mlp(self.n2(x) * (1 + sc2) + s2)
        return x


class ShapeDiT(nn.Module):
    def __init__(self, res: int = 32, patch: int = 4, d_model: int = 256, layers: int = 8, heads: int = 8):
        super().__init__()
        assert res % patch == 0
        self.res, self.patch, self.d_model = res, patch, d_model
        g = res // patch
        self.embed = nn.Conv3d(3, d_model, patch, stride=patch)
        self.register_buffer("pos", sincos_3d(d_model, g)[None], persistent=False)
        self.t_embed = TimeEmbed(d_model)
        self.blocks = nn.ModuleList([Block(d_model, heads) for _ in range(layers)])
        self.n_out = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.out = nn.Linear(d_model, patch ** 3)
        for m in (self.ada_out[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.embed.weight.view(d_model, -1))

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t, cond, mask: (B,1,R,R,R); t: (B,) -> velocity (B,1,R,R,R)."""
        B = x_t.shape[0]
        h = self.embed(torch.cat([x_t, cond, mask], 1)).flatten(2).transpose(1, 2) + self.pos.to(x_t.dtype)
        c = self.t_embed(t)
        for blk in self.blocks:
            h = blk(h, c)
        s, sc = self.ada_out(c)[:, None].chunk(2, -1)
        h = self.out(self.n_out(h) * (1 + sc) + s)                       # (B, g^3, p^3)
        g, p = self.res // self.patch, self.patch
        h = h.reshape(B, g, g, g, p, p, p).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, 1, self.res, self.res, self.res)
        return h

    def config(self) -> dict:
        return dict(res=self.res, patch=self.patch, d_model=self.d_model, layers=len(self.blocks),
                    heads=self.blocks[0].h)


class FlowMatching:
    def __init__(self, hidden_weight: float = 2.0, observed_weight: float = 0.25):
        self.hw, self.ow = hidden_weight, observed_weight

    def sample_t(self, n: int, device) -> torch.Tensor:
        return torch.sigmoid(torch.randn(n, device=device))               # logit-normal: more weight in the middle

    def loss(self, model: ShapeDiT, x1: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x1.shape[0]
        t = self.sample_t(B, x1.device)
        x0 = torch.randn_like(x1)
        tt = t.view(B, 1, 1, 1, 1)
        x_t = (1 - tt) * x0 + tt * x1
        v = model(x_t, cond, mask, t)
        w = torch.where(mask > 0.5, torch.full_like(mask, self.ow), torch.full_like(mask, self.hw))
        return (w * (v - (x1 - x0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, model: ShapeDiT, cond: torch.Tensor, mask: torch.Tensor, steps: int = 32, seed: int = 0,
               heun: bool = True) -> torch.Tensor:
        """Generate the clean TSDF given observations.  Observed voxels follow the known path exactly (data
        consistency); the model only decides the hidden volume."""
        g = torch.Generator(device=cond.device).manual_seed(int(seed))
        x = torch.randn(cond.shape, generator=g, device=cond.device)
        n_obs = torch.randn(cond.shape, generator=g, device=cond.device)
        m = mask > 0.5

        def known(t):
            return (1 - t) * n_obs + t * cond

        ts = torch.linspace(0, 1, steps + 1, device=cond.device)
        x = torch.where(m, known(ts[0]), x)
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            tb = t0.expand(cond.shape[0])
            v = model(x, cond, mask, tb)
            xn = x + (t1 - t0) * v
            if heun and i < steps - 1:
                v2 = model(xn, cond, mask, t1.expand(cond.shape[0]))
                xn = x + (t1 - t0) * 0.5 * (v + v2)
            x = torch.where(m, known(t1), xn)
        return x.clamp(-1, 1)
