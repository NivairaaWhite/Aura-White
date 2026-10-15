from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import AuraConfig


class TriplaneTokenizer(nn.Module):
    """Learned triplane token embeddings: 3 planes of (C, P, P) flattened into one sequence."""

    def __init__(self, cfg: AuraConfig):
        super().__init__()
        self.p, self.c = cfg.plane_size, cfg.plane_channels
        self.embeddings = nn.Parameter(torch.randn(3, self.c, self.p, self.p) / math.sqrt(self.c))

    def forward(self, batch: int) -> torch.Tensor:
        t = self.embeddings.permute(1, 0, 2, 3).reshape(self.c, 3 * self.p * self.p)
        return t.unsqueeze(0).expand(batch, -1, -1).contiguous()

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        return tokens.reshape(b, self.c, 3, self.p, self.p).permute(0, 2, 1, 3, 4)


class TriplaneUpsampler(nn.Module):
    def __init__(self, cfg: AuraConfig):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(cfg.plane_channels, cfg.triplane_out_channels, kernel_size=2, stride=2)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        b, _, ci, h, w = planes.shape
        out = self.upsample(planes.reshape(b * 3, ci, h, w))
        return out.reshape(b, 3, out.shape[1], out.shape[2], out.shape[3])


class NeRFMLP(nn.Module):
    """Tiny decoder: concatenated triplane features -> (density, rgb-logits)."""

    def __init__(self, cfg: AuraConfig):
        super().__init__()
        n = cfg.mlp_neurons
        layers = [nn.Linear(3 * cfg.triplane_out_channels, n), nn.SiLU()]
        for _ in range(cfg.mlp_hidden_layers - 1):
            layers += [nn.Linear(n, n), nn.SiLU()]
        layers += [nn.Linear(n, 4)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        y = self.layers(x)
        return y[..., 0:1], y[..., 1:4]
