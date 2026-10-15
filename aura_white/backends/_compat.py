"""Locating the third-party model repositories and importing them defensively."""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import types
from typing import Callable, Optional

# packages that heavy repos import at module level but only use for features Aura White never touches
_STUBBABLE = {"pymeshlab", "pythreejs", "wandb", "rembg", "matplotlib", "nvdiffrast", "xatlas", "cv2",
              "pytorch3d", "kornia", "open3d", "onnxruntime", "gradio", "fastapi", "uvicorn"}


class _StubModule(types.ModuleType):
    """Stands in for an optional package. Importing works; *using* it raises a clear error."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []  # allows `import name.sub`
        self.__aura_stub__ = True

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        if item == "RasterizeCudaContext":
            return lambda *a, **k: object()          # InstantMesh builds one; mesh extraction never rasterises with it
        return _Dummy(f"{self.__name__}.{item}")


class _Dummy:
    def __init__(self, name):
        self._n = name

    def __call__(self, *a, **k):
        raise RuntimeError(f"'{self._n}' is not available: the optional package is not installed")

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return _Dummy(f"{self._n}.{item}")


def apply_transformers_shims():
    """Newer `transformers` dropped two tiny helpers that InstantMesh's vendored ViT still imports.
    Re-adding them (they are pure functions) keeps InstantMesh working across transformers versions."""
    try:
        import torch
        import transformers.pytorch_utils as pu
    except Exception:
        return
    try:                                            # transformers >= 5 dropped PreTrainedModel.get_head_mask
        from transformers import PreTrainedModel

        if not hasattr(PreTrainedModel, "get_head_mask"):
            def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                if head_mask.dim() == 1:
                    head_mask = head_mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask[:, None, :, None, None]
                return head_mask.to(dtype=self.dtype)

            PreTrainedModel.get_head_mask = get_head_mask
    except Exception:
        pass
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not hasattr(pu, "prune_linear_layer"):
        def prune_linear_layer(layer, index, dim=0):
            index = index.to(layer.weight.device)
            W = layer.weight.index_select(dim, index).detach().clone()
            b = layer.bias.detach().clone() if dim == 1 else (layer.bias[index].detach().clone() if layer.bias is not None else None)
            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
            new.weight.requires_grad = False
            new.weight.copy_(W.contiguous())
            new.weight.requires_grad = True
            if layer.bias is not None:
                new.bias.requires_grad = False
                new.bias.copy_(b.contiguous())
                new.bias.requires_grad = True
            return new

        pu.prune_linear_layer = prune_linear_layer


_ACTIVE_STUBS: set = set()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves `import stubbed_pkg.submodule` from stub modules."""

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in _ACTIVE_STUBS:
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def _install_stub(top: str):
    _ACTIVE_STUBS.add(top)
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.append(_StubFinder())
    sys.modules[top] = _StubModule(top)


def import_with_stubs(module: str, log: Optional[Callable[[str], None]] = None, max_stubs: int = 12):
    """import_module(module); when a *stubbable* optional dependency is missing, stub it and retry."""
    apply_transformers_shims()
    for _ in range(max_stubs + 1):
        try:
            return importlib.import_module(module)
        except ModuleNotFoundError as e:
            top = (e.name or "").split(".")[0]
            if top not in _STUBBABLE or top in sys.modules and getattr(sys.modules[top], "__aura_stub__", False):
                raise
            _install_stub(top)
            for k in [k for k in sys.modules if k == module or k.startswith(module.split(".")[0] + ".")]:
                if k != top:
                    sys.modules.pop(k, None)     # drop half-imported modules so the retry starts clean
            if log:
                log(f"optional package '{top}' is missing - continuing without it")
    raise ImportError(f"could not import {module}")


class prepend_path:
    """Context manager: temporarily put a repo directory first on sys.path."""

    def __init__(self, *paths):
        self.paths = [str(p) for p in paths]

    def __enter__(self):
        for p in reversed(self.paths):
            sys.path.insert(0, p)

    def __exit__(self, *exc):
        for p in self.paths:
            try:
                sys.path.remove(p)
            except ValueError:
                pass
