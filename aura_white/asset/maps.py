"""Game-ready surface maps derived from the albedo atlas (NumPy + SciPy only).

The artwork has its lighting painted in, so the maps are deliberately gentle: a normal map from the luminance relief
(folds, bricks, rock facets pick up the light a little), an occlusion term from the cavity (luminance darker than its
surroundings), constant roughness / metallic, and an optional emissive map from very bright, saturated texels
(crystals, glow, lava).  glTF conventions: normal green = image-up, ORM = R occlusion / G roughness / B metallic.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

KIND_MATERIAL = {                       # roughness, metallic, normal strength, occlusion strength
    "character": (0.75, 0.0, 0.5, 0.35),
    "prop": (0.45, 0.35, 0.9, 0.45),
    "arena": (0.8, 0.0, 1.2, 0.5),
    "island": (0.85, 0.0, 1.2, 0.5),
    "building": (0.8, 0.0, 1.2, 0.5),
    "sphere": (0.9, 0.0, 0.8, 0.0),
}


def luminance(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    return (0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]) / 255.0


def normal_from_albedo(albedo: np.ndarray, strength: float = 1.0, blur: float = 1.0) -> np.ndarray:
    """Tangent-space normal map (uint8 RGB) from the luminance relief of the albedo."""
    from scipy import ndimage as ndi
    h = luminance(albedo)
    if blur > 0:
        h = ndi.gaussian_filter(h, blur)
    hy, hx = np.gradient(h)                                  # hy: along image-down
    s = float(strength) * 3.0
    n = np.stack([-hx * s, hy * s, np.ones_like(h)], -1)    # (R=+u, G=image-up, B=out): see the derivation in the docs
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.clip((n * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def occlusion_from_albedo(albedo: np.ndarray, strength: float = 0.4, radius: float = 6.0) -> np.ndarray:
    from scipy import ndimage as ndi
    h = luminance(albedo)
    cav = np.clip(ndi.gaussian_filter(h, radius) - h, 0.0, None)
    cav = cav / max(float(np.percentile(cav, 99.0)), 1e-3)
    return np.clip(1.0 - strength * np.clip(cav, 0.0, 1.0), 0.0, 1.0)


def orm_map(albedo: np.ndarray, roughness: float, metallic: float, occlusion: float,
            metallic_mask: Optional[np.ndarray] = None, cavity: Optional[np.ndarray] = None) -> np.ndarray:
    """ORM atlas.  `cavity` (0..1, 1 = open) is a geometric occlusion term baked from the full-resolution mesh; it is
    combined with the albedo-derived cavity."""
    ao = occlusion_from_albedo(albedo, occlusion) if occlusion > 0 else np.ones(albedo.shape[:2], np.float32)
    if cavity is not None:
        ao = np.clip(ao * (1.0 - float(occlusion or 0.5) * (1.0 - cavity)), 0.0, 1.0)
    rough = np.full(ao.shape, roughness, np.float32)
    met = np.full(ao.shape, metallic, np.float32) if metallic_mask is None else metallic * metallic_mask
    out = np.stack([ao, rough, met], -1)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def emissive_map(albedo: np.ndarray, threshold: float = 0.82, saturation: float = 0.35) -> Optional[np.ndarray]:
    """Very bright, saturated texels (glowing crystals ...) -> emissive texture; None when there are none."""
    f = albedo.astype(np.float32) / 255.0
    mx, mn = f.max(-1), f.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    m = (mx >= threshold) & (sat >= saturation)
    if m.mean() < 1e-4:
        return None
    from scipy import ndimage as ndi
    soft = ndi.gaussian_filter(m.astype(np.float32), 1.0)
    return np.clip(albedo.astype(np.float32) * soft[..., None] + 0.5, 0, 255).astype(np.uint8)
