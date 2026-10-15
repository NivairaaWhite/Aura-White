"""DINO ViT-B/16 image encoder in plain PyTorch.

Parameter names mirror the HuggingFace ViTModel layout that the TripoSR checkpoint was saved
with, so the released weights load 1:1 - but nothing here imports `transformers`, so a
transformers upgrade can never break loading again.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._compat import sdpa
from ..config import AuraConfig


class _Projection(nn.Module):
    def __init__(self, channels: int, hidden: int, patch: int):
        super().__init__()
        self.projection = nn.Conv2d(channels, hidden, kernel_size=patch, stride=patch)


class _Embeddings(nn.Module):
    def __init__(self, cfg: AuraConfig):
        super().__init__()
        d, g = cfg.enc_hidden, cfg.enc_pos_grid
        self.patch = cfg.enc_patch
        self.grid = g
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1 + g * g, d))
        self.patch_embeddings = _Projection(3, d, cfg.enc_patch)
        self._pos_cache: dict = {}

    def _positions(self, h: int, w: int, dtype) -> torch.Tensor:
        pe = self.position_embeddings
        g = self.grid
        if h == g and w == g:
            return pe
        key = (h, w, pe.dtype, pe.device)
        hit = self._pos_cache.get(key)
        if hit is not None and not torch.is_grad_enabled():
            return hit
        d = pe.shape[-1]
        cls_pos, grid_pos = pe[:, :1], pe[:, 1:]
        grid_pos = grid_pos.reshape(1, g, g, d).permute(0, 3, 1, 2)
        grid_pos = F.interpolate(
            grid_pos.float(), size=(h, w), mode="bicubic", align_corners=False
        ).to(pe.dtype)
        grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, h * w, d)
        out = torch.cat([cls_pos, grid_pos], dim=1)
        if not torch.is_grad_enabled():
            self._pos_cache[key] = out
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, height, width = x.shape
        tok = self.patch_embeddings.projection(x).flatten(2).transpose(1, 2)
        tok = torch.cat([self.cls_token.expand(b, -1, -1).to(tok.dtype), tok], dim=1)
        return tok + self._positions(height // self.patch, width // self.patch, tok.dtype)


class _SelfAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)


class _Dense(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.dense = nn.Linear(i, o)


class _Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.attention = _SelfAttention(d)
        self.output = _Dense(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads
        s = self.attention
        q = s.query(x).view(b, n, h, d // h).transpose(1, 2)
        k = s.key(x).view(b, n, h, d // h).transpose(1, 2)
        v = s.value(x).view(b, n, h, d // h).transpose(1, 2)
        o = sdpa(q, k, v).transpose(1, 2).reshape(b, n, d)
        return self.output.dense(o)


class _Layer(nn.Module):
    def __init__(self, cfg: AuraConfig):
        super().__init__()
        d = cfg.enc_hidden
        self.attention = _Attention(d, cfg.enc_heads)
        self.intermediate = _Dense(d, cfg.enc_mlp)
        self.output = _Dense(cfg.enc_mlp, d)
        self.layernorm_before = nn.LayerNorm(d, eps=cfg.enc_eps)
        self.layernorm_after = nn.LayerNorm(d, eps=cfg.enc_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.layernorm_before(x)) + x
        y = self.output.dense(F.gelu(self.intermediate.dense(self.layernorm_after(x))))
        return y + x


class _Encoder(nn.Module):
    def __init__(self, cfg: AuraConfig):
        super().__init__()
        self.layer = nn.ModuleList([_Layer(cfg) for _ in range(cfg.enc_layers)])


class DinoViT(nn.Module):
    def __init__(self, cfg: AuraConfig):
        super().__init__()
        self.embeddings = _Embeddings(cfg)
        self.encoder = _Encoder(cfg)
        self.layernorm = nn.LayerNorm(cfg.enc_hidden, eps=cfg.enc_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embeddings(x)
        for blk in self.encoder.layer:
            h = blk(h)
        return self.layernorm(h)


class ImageTokenizer(nn.Module):
    """(B,3,H,W) in [0,1]  ->  (B, 1 + (H/16)*(W/16), enc_hidden)."""

    def __init__(self, cfg: AuraConfig):
        super().__init__()
        self.model = DinoViT(cfg)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images - self.image_mean.to(images.dtype)) / self.image_std.to(images.dtype)
        return self.model(x)
