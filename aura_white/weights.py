"""Finding, downloading and converting weights.

* Re-uses a TripoSR `model.ckpt` you already have (HuggingFace cache) - no second 1.7 GB download.
* Converts once to a compact, pickle-free .safetensors file (fp16 by default: half the size).
* Uses only the Python standard library for the download (no huggingface_hub / requests).
* Accepts checkpoints written with either the old or the new HuggingFace ViT naming.
"""
from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from ._compat import cache_home, load_state_dict_file

OFFICIAL_URL = "https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt"
FP16_NAME = "aura_white_fp16.safetensors"
FP32_NAME = "aura_white_fp32.safetensors"

Progress = Optional[Callable[[str, float], None]]


def weights_dir() -> Path:
    d = cache_home() / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------------------------
# key normalisation
# --------------------------------------------------------------------------------------------
# Newer `transformers` releases renamed the ViT sub-modules. The public checkpoint uses the old
# names, and Aura White's own layout uses the old names too, so map the new ones back.
_RENAMES = [
    (re.compile(r"\.layers\.(\d+)\.attention\.q_proj\."), r".encoder.layer.\1.attention.attention.query."),
    (re.compile(r"\.layers\.(\d+)\.attention\.k_proj\."), r".encoder.layer.\1.attention.attention.key."),
    (re.compile(r"\.layers\.(\d+)\.attention\.v_proj\."), r".encoder.layer.\1.attention.attention.value."),
    (re.compile(r"\.layers\.(\d+)\.attention\.o_proj\."), r".encoder.layer.\1.attention.output.dense."),
    (re.compile(r"\.layers\.(\d+)\.mlp\.fc1\."), r".encoder.layer.\1.intermediate.dense."),
    (re.compile(r"\.layers\.(\d+)\.mlp\.fc2\."), r".encoder.layer.\1.output.dense."),
    (re.compile(r"\.layers\.(\d+)\.(layernorm_before|layernorm_after)\."), r".encoder.layer.\1.\2."),
]


def normalize_state_dict(sd: dict) -> dict:
    """Rename new-style ViT keys, drop the unused ViT pooler and any wrapper prefixes."""
    out = {}
    strip = bool(sd) and all(k.startswith("model.") for k in sd)  # e.g. a Lightning-style wrapper
    for k, v in sd.items():
        if strip:
            k = k[6:]
        if ".pooler." in k:
            continue
        for pat, rep in _RENAMES:
            k = pat.sub(rep, k)
        out[k] = v
    return out


# --------------------------------------------------------------------------------------------
# locating an existing TripoSR download
# --------------------------------------------------------------------------------------------
def _hf_hub_roots() -> list:
    roots = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def find_existing_triposr() -> Optional[Path]:
    """Look for an already-downloaded stabilityai/TripoSR checkpoint on this machine."""
    for root in _hf_hub_roots():
        snaps = root / "models--stabilityai--TripoSR" / "snapshots"
        try:
            if not snaps.is_dir():
                continue
            for name in ("model.safetensors", "model.ckpt"):
                for f in sorted(snaps.glob(f"*/{name}")):
                    if f.exists():
                        return f
        except OSError:
            continue
    for p in (Path.cwd() / "model.ckpt", Path.cwd() / "weights" / "model.ckpt"):
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------------------------
# download (stdlib only, resumable)
# --------------------------------------------------------------------------------------------
def download(url: str, dest: Path, progress: Progress = None, timeout: int = 60) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(6):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "aura-white/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status = getattr(r, "status", 200)
                length = int(r.headers.get("Content-Length") or 0)
                if status == 200 and have:  # server ignored Range -> restart cleanly
                    have = 0
                total = have + length if length else 0
                mode = "ab" if have else "wb"
                done = have
                t0 = time.time()
                with open(part, mode) as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if progress and total:
                            mbs = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            progress(f"downloading weights {done / 1e9:.2f}/{total / 1e9:.2f} GB ({mbs:.1f} MB/s)", done / total)
                if total and done < total:
                    raise urllib.error.URLError("connection closed early")
            os.replace(part, dest)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():  # already complete
                os.replace(part, dest)
                return dest
            if e.code in (401, 403, 404):
                raise RuntimeError(f"Download refused ({e.code}) for {url}. Download model.ckpt manually and pass --weights.") from e
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download {url} after several attempts. Download it in a browser and pass --weights PATH.")


# --------------------------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------------------------
def convert(src: Path, dst: Optional[Path] = None, dtype: str = "fp16", trust_pickle: bool = False) -> Path:
    """Convert any supported checkpoint to a clean .safetensors file (or .pt without safetensors)."""
    import torch

    sd = normalize_state_dict(load_state_dict_file(src, trust_pickle=trust_pickle))
    cast = torch.float16 if dtype == "fp16" else None
    clean = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        v = v.detach().cpu()
        if cast is not None and v.is_floating_point():
            v = v.to(cast)
        clean[k] = v.contiguous().clone()
    dst = Path(dst) if dst else weights_dir() / (FP16_NAME if dtype == "fp16" else FP32_NAME)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file(clean, str(dst), metadata={"format": "pt", "producer": "aura-white"})
    except ImportError:
        dst = dst.with_suffix(".pt")
        torch.save(clean, str(dst))
    return dst


# --------------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------------
def resolve_weights(explicit: Optional[str] = None, allow_download: bool = True, progress: Progress = None,
                    trust_pickle: bool = False, dtype: str = "fp16") -> Path:
    """Return a path to loadable weights, preparing them if needed."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"weights file not found: {p}")
        return p
    env = os.environ.get("AURA_WHITE_WEIGHTS")
    if env and Path(env).is_file():
        return Path(env)

    wd = weights_dir()
    for name in (FP16_NAME, FP32_NAME, Path(FP16_NAME).with_suffix(".pt").name):
        if (wd / name).is_file():
            return wd / name

    src = find_existing_triposr()
    if src is None:
        if not allow_download:
            raise FileNotFoundError("No weights found. Run `aura-white fetch` (about 1.7 GB, one time).")
        url = os.environ.get("AURA_WHITE_WEIGHTS_URL", OFFICIAL_URL)
        src = download(url, wd / "model.ckpt", progress)
    if progress:
        progress("converting weights to compact format", 0.0)
    try:
        out = convert(src, dtype=dtype, trust_pickle=trust_pickle)
    except Exception:
        # Conversion is an optimisation, never a requirement: fall back to reading the original.
        return src
    if progress:
        progress("weights ready", 1.0)
    return out
