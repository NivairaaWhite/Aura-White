"""Hardware probing and memory planning.

The planner answers one question: *given this machine, how do I run a model of this size without running out of
memory?*  It never answers *which model should I use* - that is the recipe's job - so the same picture gives the same
mesh on a 4 GB laptop GPU and on a 80 GB server card, only slower on the former.

Strategies, cheapest first (all numerically equivalent up to float noise):

  none        everything resident on the GPU
  model       whole sub-models (text encoder, denoiser, decoder) are moved to the GPU only while they run
  sequential  individual layers are streamed in; only the layer being executed is resident
  cpu         run on the processor (always possible, slow)

`run_with_oom_retry` turns the estimates into a guarantee: if a stage still runs out of memory, the plan is
escalated (more offloading, smaller chunks) and the stage re-run, instead of failing or - worse - quietly dropping to
a lower-quality method.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Tuple, TypeVar

T = TypeVar("T")
OFFLOAD_ORDER = ("none", "model", "sequential", "cpu")
MIN_CHUNK_SCALE = 1.0 / 16.0


@dataclass(frozen=True)
class Hardware:
    device: str = "cpu"                   # cuda | mps | cpu
    name: str = "cpu"
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    bf16: bool = False
    fp16: bool = False

    def describe(self) -> str:
        if self.device == "cuda":
            return f"{self.name}: {self.vram_free_gb:.1f} of {self.vram_total_gb:.1f} GB GPU memory free, {self.ram_free_gb:.0f} GB RAM free"
        return f"{self.name} ({self.device}), {self.ram_free_gb:.0f} GB RAM free"


def _ram_gb() -> Tuple[float, float]:
    try:
        import psutil                                    # optional
        m = psutil.virtual_memory()
        return m.total / 1024 ** 3, m.available / 1024 ** 3
    except Exception:
        pass
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                info[k] = float(v.split()[0]) / 1024 ** 2
        return info.get("MemTotal", 0.0), info.get("MemAvailable", info.get("MemFree", 0.0))
    except Exception:
        pass
    try:
        page, n, avail = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_AVPHYS_PAGES")
        return page * n / 1024 ** 3, page * avail / 1024 ** 3
    except Exception:
        return 0.0, 0.0


def probe_hardware(device: str = "auto") -> Hardware:
    """What this machine can do right now.  Never raises: a missing torch simply means 'cpu'."""
    ram_t, ram_f = _ram_gb()
    try:
        import torch
    except Exception:
        return Hardware("cpu", "cpu (torch not installed)", ram_total_gb=ram_t, ram_free_gb=ram_f)
    want = device if device != "auto" else None
    try:
        if want in (None, "cuda") and torch.cuda.is_available():
            idx = 0
            free, total = torch.cuda.mem_get_info(idx)
            cap = torch.cuda.get_device_capability(idx)
            return Hardware("cuda", torch.cuda.get_device_name(idx), total / 1024 ** 3, free / 1024 ** 3, ram_t, ram_f,
                            bf16=bool(getattr(torch.cuda, "is_bf16_supported", lambda: cap[0] >= 8)()), fp16=cap[0] >= 6)
        mps = getattr(torch.backends, "mps", None)
        if want in (None, "mps") and mps is not None and mps.is_available():
            return Hardware("mps", "Apple GPU (unified memory)", ram_t, ram_f, ram_t, ram_f, bf16=False, fp16=True)
    except Exception:
        pass
    return Hardware("cpu", "cpu", 0.0, 0.0, ram_t, ram_f, bf16=True, fp16=False)


@dataclass(frozen=True)
class StageCost:
    """Rough memory footprint of one model stage (GB, half precision).  These are *estimates for choosing a strategy*;
    the OOM retry corrects any that are too optimistic."""
    name: str
    weights_gb: float                     # all weights, fp16
    largest_part_gb: float                # the biggest sub-model that must be resident while it runs
    activations_gb: float                 # working memory at chunk_scale = 1
    layer_gb: float = 0.4                 # the biggest single layer (what sequential offload needs resident)

    def peak(self, mode: str, chunk_scale: float) -> float:
        act = self.activations_gb * chunk_scale
        if mode == "none":
            return self.weights_gb + act
        if mode == "model":
            return self.largest_part_gb + act
        if mode == "sequential":
            return self.layer_gb + act
        return 0.0


# Footprints of the stages Aura White drives (fp16 weights; activations at the default chunk size).
STAGES: Dict[str, StageCost] = {s.name: s for s in (
    StageCost("hunyuan3d2_shape", 3.1, 2.3, 2.6, 0.12),        # 1.1B DiT + DINOv2-L + shape VAE; the vendor quotes ~6 GB total
    StageCost("hunyuan3d2mv_shape", 3.1, 2.3, 2.8, 0.12),
    StageCost("hunyuan3d2mini_shape", 1.9, 1.3, 2.2, 0.10),
    StageCost("aura_triposr", 1.7, 1.2, 3.2, 0.10),            # built-in TripoSR-compatible network
    StageCost("sdxl_text2image", 7.0, 5.2, 1.6, 0.25),
    StageCost("sdxl_depth_paint", 12.0, 5.2, 1.8, 0.25),       # SDXL + depth ControlNet + IP-Adapter (+ its image encoder)
    StageCost("flux_text2image", 23.8, 23.8, 2.0, 0.55),
)}


@dataclass(frozen=True)
class ExecPlan:
    stage: str
    device: str
    offload: str = "none"
    dtype: str = "float16"
    chunk_scale: float = 1.0
    est_peak_gb: float = 0.0
    feasible: bool = True
    note: str = ""

    def escalate(self, allow_cpu: bool = True) -> Optional["ExecPlan"]:
        """The next, safer plan - or None when nothing safer exists."""
        if self.chunk_scale > MIN_CHUNK_SCALE + 1e-9 and self.offload != "cpu":
            return replace(self, chunk_scale=max(MIN_CHUNK_SCALE, self.chunk_scale / 2.0),
                           note=f"chunks halved after out-of-memory ({self.offload})")
        i = OFFLOAD_ORDER.index(self.offload)
        nxt = OFFLOAD_ORDER[i + 1] if i + 1 < len(OFFLOAD_ORDER) else None
        if nxt is None or (nxt == "cpu" and not allow_cpu):
            return None
        if nxt == "cpu":
            return replace(self, device="cpu", offload="cpu", dtype="float32", chunk_scale=1.0,
                           note="falling back to the processor after out-of-memory")
        return replace(self, offload=nxt, chunk_scale=1.0, note=f"offload escalated to {nxt} after out-of-memory")


def plan_stage(stage: str, hw: Hardware, margin: float = 0.88, reserve_gb: float = 0.6) -> ExecPlan:
    """Choose how to run `stage` on `hw`.  Quality-neutral by construction."""
    cost = STAGES.get(stage)
    if cost is None:
        raise KeyError(f"unknown stage {stage!r}; known: {', '.join(STAGES)}")
    if hw.device == "cpu":
        need = cost.weights_gb * 2.0 * 1.15                  # fp32 on the processor
        ok = hw.ram_free_gb <= 0 or hw.ram_free_gb >= need
        return ExecPlan(stage, "cpu", "cpu", "bfloat16" if hw.bf16 else "float32", 1.0, need, ok,
                        "" if ok else f"needs about {need:.0f} GB of free RAM, {hw.ram_free_gb:.0f} GB available")
    if hw.device == "mps":
        return ExecPlan(stage, "mps", "none", "float16", 1.0, cost.weights_gb + cost.activations_gb, True, "unified memory")
    budget = max(0.0, hw.vram_free_gb * margin - reserve_gb)
    dtype = "float16" if hw.fp16 else ("bfloat16" if hw.bf16 else "float32")
    for mode in ("none", "model", "sequential"):
        scale = 1.0
        while True:
            peak = cost.peak(mode, scale)
            if peak <= budget:
                if mode in ("model", "sequential") and hw.ram_free_gb and hw.ram_free_gb < cost.weights_gb * 1.1:
                    break                                        # offloading needs the weights to fit in RAM
                note = "" if (mode == "none" and scale == 1.0) else f"{mode} offload" + (f", chunks x{scale:g}" if scale != 1.0 else "")
                return ExecPlan(stage, "cuda", mode, dtype, scale, peak, True, note)
            if scale <= MIN_CHUNK_SCALE + 1e-9 or mode == "none":
                break
            scale /= 2.0
    # nothing fits on the GPU: the processor is slower but gives the identical result
    need = cost.weights_gb * 2.0 * 1.15
    ok = hw.ram_free_gb <= 0 or hw.ram_free_gb >= need
    return ExecPlan(stage, "cpu", "cpu", "bfloat16" if hw.bf16 else "float32", 1.0, need, ok,
                    "GPU memory too small for this stage - running on the processor (same result, slower)" if ok else
                    f"neither {hw.vram_free_gb:.1f} GB of GPU memory nor {hw.ram_free_gb:.0f} GB of RAM is enough")


def is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return ("outofmemory" in name or "out of memory" in msg or "cuda error: out of memory" in msg
            or "cublas_status_alloc_failed" in msg or "mps backend out of memory" in msg or "defaultcpuallocator" in msg)


def free_accelerator_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:
        pass


def run_with_oom_retry(fn: Callable[[ExecPlan], T], plan: ExecPlan, say: Optional[Callable[[str], None]] = None,
                       allow_cpu: bool = True, max_attempts: int = 24) -> Tuple[T, ExecPlan]:
    """Call `fn(plan)`; on out-of-memory escalate the plan and call again.  Returns (result, plan that worked)."""
    say = say or (lambda *_: None)
    attempts = 0
    while True:
        attempts += 1
        try:
            return fn(plan), plan
        except BaseException as e:                           # noqa: BLE001 - re-raised unless it is a memory error
            if not is_oom(e):
                raise
            free_accelerator_memory()
            nxt = plan.escalate(allow_cpu) if attempts < max_attempts else None
            if nxt is None:
                raise MemoryError(f"{plan.stage}: out of memory even with {plan.offload} offload and the smallest chunks; "
                                  f"free memory or use a smaller recipe preset ({e})") from e
            say(f"{plan.stage}: out of memory - retrying ({nxt.note})")
            plan = nxt
