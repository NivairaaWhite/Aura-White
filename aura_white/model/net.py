from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AuraConfig
from .dino import ImageTokenizer
from .renderer import query_triplane, render_rays, spherical_cameras
from .transformer import Transformer1D
from .triplane import NeRFMLP, TriplaneTokenizer, TriplaneUpsampler


class AuraNet(nn.Module):
    """image -> triplane scene code -> (density, color) field. Attribute names are the same as
    TripoSR's so `model.ckpt` state dicts load without any key renaming."""

    def __init__(self, cfg: AuraConfig | None = None):
        super().__init__()
        self.cfg = cfg or AuraConfig()
        c = self.cfg
        self.image_tokenizer = ImageTokenizer(c)
        self.tokenizer = TriplaneTokenizer(c)
        self.backbone = Transformer1D(c)
        self.post_processor = TriplaneUpsampler(c)
        self.decoder = NeRFMLP(c)

    # ---- precision -------------------------------------------------------------------------
    def set_precision(self, dtype: torch.dtype) -> "AuraNet":
        """Encoder/backbone in `dtype` (the heavy part); the tiny density decoder stays fp32."""
        for m in (self.image_tokenizer, self.tokenizer, self.backbone, self.post_processor):
            m.to(dtype)
        self.decoder.float()
        self._enc_dtype = dtype
        return self

    # ---- inference -------------------------------------------------------------------------
    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, S, S) float in [0,1]  ->  scene codes (B, 3, C, 2P, 2P) fp32."""
        dt = getattr(self, "_enc_dtype", torch.float32)
        size = self.cfg.cond_image_size
        if images.shape[-1] != size or images.shape[-2] != size:
            try:
                images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            except (NotImplementedError, RuntimeError, TypeError):  # backend without antialias support
                images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        feats = self.image_tokenizer(images.to(dt))
        tokens = self.tokenizer(images.shape[0]).to(dt)
        tokens = self.backbone(tokens, feats)
        codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return codes.float()

    @torch.inference_mode()
    def query(self, scene_code: torch.Tensor, positions: torch.Tensor, chunk: int = 131072, want_color: bool = True):
        """scene_code: (3, C, H, W). positions: (N,3) world coords. -> density_act (N,), rgb (N,3)."""
        return query_triplane(self.cfg, self.decoder, scene_code, positions, chunk, want_color)

    @torch.inference_mode()
    def render(self, scene_code: torch.Tensor, n_views=24, elevation=0.0, distance=1.9, fovy=40.0, size=192):
        """Turntable NeRF render -> list of HxWx3 uint8 numpy arrays (preview only, slow on CPU)."""
        rays_o, rays_d = spherical_cameras(n_views, elevation, distance, fovy, size, size)
        dev = scene_code.device
        frames = []
        for i in range(n_views):
            img = render_rays(self.cfg, self.decoder, scene_code, rays_o[i].reshape(-1, 3).to(dev), rays_d[i].reshape(-1, 3).to(dev))
            frames.append((img.reshape(size, size, 3).clamp(0, 1).cpu().numpy() * 255).astype("uint8"))
        return frames


@torch.no_grad()
def demo_init(net: AuraNet, seed: int = 0) -> AuraNet:
    """Well-conditioned random weights so an untrained network still yields a non-trivial field.
    Only used by `doctor` and the tests - real use loads the trained checkpoint."""
    g = torch.Generator().manual_seed(seed)
    for name, p in net.named_parameters():
        if name.endswith(("cls_token", "position_embeddings", "tokenizer.embeddings")):
            v = 0.5 * torch.randn(p.shape, generator=g)
        elif p.ndim == 1 and name.endswith("weight") and "norm" in name:
            v = 1 + 0.1 * torch.randn(p.shape, generator=g)
        elif p.ndim >= 2:
            fan_in = max(1, int(torch.tensor(p.shape[1:]).prod()))
            v = ((0.85 if name.startswith("decoder.") else 1.6) / fan_in ** 0.5) * torch.randn(p.shape, generator=g)
        else:
            v = 0.1 * torch.randn(p.shape, generator=g)
        p.copy_(v.to(p.dtype))
    return net
