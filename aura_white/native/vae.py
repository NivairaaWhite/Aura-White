"""ShapeVAE: a coloured surface point cloud  <->  a structured latent grid  <->  a signed distance and a colour at ANY point.

Aura's own implementation of the "structured latent" idea (the family TRELLIS belongs to): the latent is a dense grid of
cells in the object's own frame, and each cell describes the geometry and colour of its own piece of space.  That
locality is what makes it learn fast from little data - a convolutional decoder never has to discover that the
geometry at (x, y, z) depends mostly on what is near (x, y, z) - and it is what lets the generator treat the latent like a
small 3-D image.  (A dense grid wastes cells on empty space; a sparse version is the planned upgrade - see NATIVE.md.)

  encode   points (B,N,9) -> scatter into a (2G)^3 voxel grid (offset, normal, colour, count) -> 3-D convs -> mu, logvar (B,C,G,G,G)
  decode   z -> 3-D convs -> feature grid (B,W,2G,2G,2G) --trilinear lookup at q + Fourier(q)--> MLP -> sdf (B,Q), rgb (B,Q,3)

Pure PyTorch, no compiled extension.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VAEConfig


class Fourier(nn.Module):
    def __init__(self, bands: int):
        super().__init__()
        self.register_buffer("freq", (2.0 ** torch.arange(bands)) * math.pi, persistent=False)
        self.out_dim = 3 + 3 * 2 * bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:                       # (...,3)
        a = x.unsqueeze(-1) * self.freq                                         # (...,3,B)
        return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


class MHA(nn.Module):
    def __init__(self, d: int, heads: int, d_kv: Optional[int] = None):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d_kv or d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = x if ctx is None else ctx
        B, N, D = x.shape
        M = ctx.shape[1]
        q = self.q(x).view(B, N, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, M, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(a.transpose(1, 2).reshape(B, N, D))


class MLP(nn.Module):
    def __init__(self, d: int, ratio: float = 4.0, d_out: Optional[int] = None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, int(d * ratio)), nn.GELU(approximate="tanh"), nn.Linear(int(d * ratio), d_out or d))

    def forward(self, x):
        return self.net(x)


class SelfBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = MHA(d, heads), MLP(d)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


def _groups(w: int) -> int:
    for g in (8, 6, 4, 3, 2, 1):
        if w % g == 0:
            return g
    return 1


class Res3d(nn.Module):
    def __init__(self, w: int):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(_groups(w), w), nn.GroupNorm(_groups(w), w)
        self.c1, self.c2 = nn.Conv3d(w, w, 3, padding=1), nn.Conv3d(w, w, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return x + self.c2(F.silu(self.n2(h)))


class ShapeVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        W, C, G, R = cfg.width, cfg.latent_dim, cfg.grid, cfg.res
        self.fourier = Fourier(cfg.fourier_bands)
        # encoder
        self.enc_in = nn.Conv3d(10, W, 3, padding=1)
        self.enc1 = nn.Sequential(*[Res3d(W) for _ in range(cfg.enc_blocks)])
        self.down = nn.Conv3d(W, W, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(*[Res3d(W) for _ in range(cfg.enc_blocks)])
        self.to_lat = nn.Conv3d(W, 2 * C, 1)
        # decoder: all the heavy convolutions run at the latent resolution; one transposed conv makes the (2G)^3 feature grid
        Wf = max(16, W // 2)
        self.feat_width = Wf
        self.from_lat = nn.Conv3d(C, W, 3, padding=1)
        self.dec1 = nn.Sequential(*[Res3d(W) for _ in range(cfg.dec_blocks)])
        self.up = nn.ConvTranspose3d(W, Wf, 2, stride=2)
        self.head = nn.Sequential(nn.Linear(Wf + self.fourier.out_dim, 2 * W), nn.GELU(approximate="tanh"),
                                  nn.Linear(2 * W, 2 * W), nn.GELU(approximate="tanh"), nn.Linear(2 * W, 4))

    # -- encoder
    def voxelize(self, pts: torch.Tensor, has_colour: Optional[torch.Tensor]) -> torch.Tensor:
        """(B,N,9) -> (B,10,R,R,R): mean in-cell offset (3), mean normal (3), mean colour (3), occupancy count (1)."""
        B, N, _ = pts.shape
        R = self.cfg.res
        x = pts[..., :3].float().clamp(-0.9999, 0.9999)
        u = (x + 1.0) * 0.5 * R
        idx = u.floor().long().clamp(0, R - 1)
        off = u - idx.to(u.dtype) - 0.5
        hc = torch.ones(B, device=pts.device) if has_colour is None else has_colour.float()
        rgb = pts[..., 6:9].float() * hc.view(B, 1, 1)
        feat = torch.cat([off, pts[..., 3:6].float(), rgb], -1)                           # (B,N,9)
        flat = idx[..., 0] * R * R + idx[..., 1] * R + idx[..., 2] + torch.arange(B, device=pts.device).view(B, 1) * R ** 3
        acc = torch.zeros(B * R ** 3, 10, device=pts.device)
        acc.index_add_(0, flat.reshape(-1), torch.cat([feat, torch.ones(B, N, 1, device=pts.device)], -1).reshape(-1, 10))
        cnt = acc[:, 9:10]
        mean = acc[:, :9] / cnt.clamp(min=1.0)
        out = torch.cat([mean, (cnt / 4.0).clamp(max=1.0)], -1)
        return out.view(B, R, R, R, 10).permute(0, 4, 1, 2, 3).contiguous()

    def encode(self, pts: torch.Tensor, has_colour: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """pts (B,N,9) = xyz, normal, rgb -> mu, logvar (B,C,G,G,G)."""
        x = self.enc_in(self.voxelize(pts, has_colour).to(self.enc_in.weight.dtype))
        x = self.enc2(self.enc1(self.down(x)))
        mu, logvar = self.to_lat(x).chunk(2, 1)
        return mu, logvar.clamp(-12.0, 4.0)

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        return mu + eps * (0.5 * logvar).exp()

    # -- decoder
    def tokens(self, z: torch.Tensor) -> torch.Tensor:
        """latent (B,C,G,G,G) -> feature grid (B,Wf,2G,2G,2G)."""
        return self.up(F.silu(self.dec1(self.from_lat(z))))

    def query(self, h: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h = tokens(z); q (B,Q,3) world points in [-1,1] -> sdf (B,Q), rgb (B,Q,3) in [0,1]."""
        B, Q, _ = q.shape
        grid = q[..., [2, 1, 0]].reshape(B, Q, 1, 1, 3).to(h.dtype)                         # grid_sample wants (z, y, x)
        f = F.grid_sample(h, grid, mode="bilinear", padding_mode="border", align_corners=False)       # (B,W,Q,1,1)
        f = f.view(B, h.shape[1], Q).transpose(1, 2)
        o = self.head(torch.cat([f, self.fourier(q).to(f.dtype)], -1))
        return o[..., 0], torch.sigmoid(o[..., 1:])

    def decode(self, z: torch.Tensor, q: torch.Tensor, chunk: int = 131072):
        h = self.tokens(z)
        if q.shape[1] <= chunk:
            return self.query(h, q)
        s, c = [], []
        for i in range(0, q.shape[1], chunk):
            a, b = self.query(h, q[:, i:i + chunk])
            s.append(a)
            c.append(b)
        return torch.cat(s, 1), torch.cat(c, 1)

    # -- training loss
    def loss(self, pts: torch.Tensor, q: torch.Tensor, sdf: torch.Tensor, rgb: torch.Tensor,
             has_colour: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        mu, logvar = self.encode(pts, has_colour)
        z = self.sample(mu, logvar)
        s_hat, c_hat = self.decode(z, q)
        t = cfg.trunc
        l_sdf = (s_hat.clamp(-t, t) - sdf.clamp(-t, t)).abs().mean() / t
        inside = (sdf < 0).float()
        l_occ = F.binary_cross_entropy_with_logits(-s_hat * (8.0 / t), inside)
        out = {"sdf": l_sdf, "occ": l_occ}
        if cfg.colour:
            near = (sdf.abs() < 0.03).float()
            if has_colour is not None:
                near = near * has_colour.view(-1, 1).to(near.dtype)
            out["rgb"] = ((c_hat - rgb).abs().mean(-1) * near).sum() / near.sum().clamp(min=1.0)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        out["kl"] = kl
        out["total"] = l_sdf + 0.2 * l_occ + 0.5 * out.get("rgb", 0.0) + cfg.kl_weight * kl
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
