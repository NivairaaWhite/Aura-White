"""Model presets for Aura Native.  `tiny` runs on a laptop CPU in minutes (tests, demos, the self-improvement experiment);
s / m / l are the sizes meant for a real GPU.  Parameter counts are computed from the code (`aura-white native budget`),
never quoted from memory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict


@dataclass
class VAEConfig:
    """Structured-latent VAE: the latent is a dense grid of `grid`^3 cells x `latent_dim` channels in the object's own frame
    (every cell describes the geometry and colour of its own piece of space - local, so it learns fast from little data).
    Encoder: surface points -> a (2*grid)^3 voxel feature grid -> 3-D convolutions -> latent.  Decoder: latent -> 3-D
    convolutions -> feature grid -> trilinear lookup + small MLP -> signed distance and colour at ANY point."""
    grid: int = 8                 # latent cells per side
    latent_dim: int = 8           # channels per cell
    width: int = 48               # convolution width
    enc_blocks: int = 1
    dec_blocks: int = 2
    fourier_bands: int = 6
    n_points: int = 1024          # surface points fed to the encoder
    n_queries: int = 2048         # SDF queries per shape per training step
    kl_weight: float = 1e-4
    trunc: float = 0.1            # SDF truncation (world units) for the regression loss
    colour: bool = True

    @property
    def res(self) -> int:         # feature-grid resolution (the decoder upsamples once)
        return 2 * self.grid


@dataclass
class FlowConfig:
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    patch3: int = 2               # latent cells grouped per token side: tokens = (grid / patch3)^3
    image_size: int = 64
    patch: int = 8
    img_layers: int = 2           # blocks in the image encoder
    max_views: int = 4
    cond_drop: float = 0.1        # classifier-free-guidance dropout while training
    mlp_ratio: float = 4.0


@dataclass
class NativeConfig:
    name: str = "tiny"
    vae: VAEConfig = field(default_factory=VAEConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)

    def as_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "NativeConfig":
        return NativeConfig(d.get("name", "custom"), VAEConfig(**d["vae"]), FlowConfig(**d["flow"]))


PRESETS: Dict[str, NativeConfig] = {
    "tiny": NativeConfig("tiny"),
    # a real GPU, narrow-domain student (game props / characters / weapons): the size to train first
    "s": NativeConfig("s",
                      VAEConfig(grid=16, latent_dim=8, width=96, enc_blocks=2, dec_blocks=3, fourier_bands=8, n_points=8192,
                                n_queries=16384),
                      FlowConfig(d_model=512, layers=12, heads=8, patch3=2, image_size=224, patch=14, img_layers=4)),
    "m": NativeConfig("m",
                      VAEConfig(grid=24, latent_dim=8, width=128, enc_blocks=2, dec_blocks=4, fourier_bands=8, n_points=16384,
                                n_queries=32768),
                      FlowConfig(d_model=768, layers=16, heads=12, patch3=2, image_size=224, patch=14, img_layers=6)),
    "l": NativeConfig("l",
                      VAEConfig(grid=32, latent_dim=16, width=192, enc_blocks=3, dec_blocks=4, fourier_bands=10, n_points=32768,
                                n_queries=65536),
                      FlowConfig(d_model=1024, layers=24, heads=16, patch3=2, image_size=224, patch=14, img_layers=8)),
}


def preset(name: str) -> NativeConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose one of {', '.join(PRESETS)}")
    return NativeConfig.from_dict(PRESETS[name].as_dict())
