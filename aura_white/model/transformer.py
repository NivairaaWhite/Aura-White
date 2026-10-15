"""Triplane transformer backbone (16 x [self-attn, cross-attn, GEGLU feed-forward]).
Parameter names match the released checkpoint; no `diffusers` code is needed."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._compat import sdpa
from ..config import AuraConfig


class _Attn(nn.Module):
    def __init__(self, q_dim: int, heads: int, head_dim: int, kv_dim: int | None = None):
        super().__init__()
        inner = heads * head_dim
        self.heads, self.head_dim = heads, head_dim
        self.to_q = nn.Linear(q_dim, inner, bias=False)
        self.to_k = nn.Linear(kv_dim or q_dim, inner, bias=False)
        self.to_v = nn.Linear(kv_dim or q_dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, q_dim)])

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        ctx = x if ctx is None else ctx
        b, n, _ = x.shape
        h, d = self.heads, self.head_dim
        q = self.to_q(x).view(b, n, h, d).transpose(1, 2)
        k = self.to_k(ctx).view(b, -1, h, d).transpose(1, 2)
        v = self.to_v(ctx).view(b, -1, h, d).transpose(1, 2)
        o = sdpa(q, k, v).transpose(1, 2).reshape(b, n, h * d)
        return self.to_out[0](o)


class _GEGLU(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.proj = nn.Linear(i, o * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, gate = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(gate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # index 1 is where the original keeps a (parameter-free) Dropout; kept so names line up
        self.net = nn.ModuleList([_GEGLU(dim, dim * 4), nn.Identity(), nn.Linear(dim * 4, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[0](x))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, cross_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = _Attn(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = _Attn(dim, heads, head_dim, kv_dim=cross_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), ctx) + x
        return self.ff(self.norm3(x)) + x


class Transformer1D(nn.Module):
    def __init__(self, cfg: AuraConfig):
        super().__init__()
        c = cfg.plane_channels
        inner = cfg.bb_heads * cfg.bb_head_dim
        self.norm = nn.GroupNorm(cfg.bb_norm_groups, c, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(c, inner)
        self.transformer_blocks = nn.ModuleList(
            [_Block(inner, cfg.bb_heads, cfg.bb_head_dim, cfg.enc_hidden) for _ in range(cfg.bb_layers)]
        )
        self.proj_out = nn.Linear(inner, c)

    def forward(self, tokens: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """tokens: (B, C, N)   ctx: (B, M, enc_hidden)   ->   (B, C, N)"""
        residual = tokens
        x = self.norm(tokens).permute(0, 2, 1)
        x = self.proj_in(x)
        for blk in self.transformer_blocks:
            x = blk(x, ctx)
        x = self.proj_out(x).permute(0, 2, 1)
        return x + residual
