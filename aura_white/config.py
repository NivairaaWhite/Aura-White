"""Model hyper-parameters. Defaults reproduce the released TripoSR checkpoint exactly
(values taken from stabilityai/TripoSR config.yaml), so its weights load unchanged."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AuraConfig:
    cond_image_size: int = 512

    # --- image encoder: DINO ViT-B/16 ------------------------------------------------------
    enc_hidden: int = 768
    enc_layers: int = 12
    enc_heads: int = 12
    enc_mlp: int = 3072
    enc_patch: int = 16
    enc_pos_grid: int = 14  # pretrained position table is 14x14 (224px / 16); resized on the fly
    enc_eps: float = 1e-12

    # --- triplane tokens + transformer backbone --------------------------------------------
    plane_size: int = 32
    plane_channels: int = 1024
    bb_heads: int = 16
    bb_head_dim: int = 64
    bb_layers: int = 16
    bb_norm_groups: int = 32

    # --- triplane upsampler + NeRF MLP ------------------------------------------------------
    triplane_out_channels: int = 40
    mlp_neurons: int = 64
    mlp_hidden_layers: int = 9

    # --- renderer ----------------------------------------------------------------------------
    radius: float = 0.87
    density_bias: float = -1.0
    samples_per_ray: int = 128

    @classmethod
    def tiny(cls) -> "AuraConfig":
        """A few-million-parameter variant with random weights. Used by `doctor` and the tests
        to exercise the whole pipeline in seconds without downloading anything."""
        return cls(
            cond_image_size=64, enc_hidden=64, enc_layers=2, enc_heads=4, enc_mlp=128,
            enc_pos_grid=4, plane_size=8, plane_channels=64, bb_heads=4, bb_head_dim=16,
            bb_layers=2, bb_norm_groups=8, triplane_out_channels=8, mlp_neurons=32,
            mlp_hidden_layers=3, samples_per_ray=32,
        )
