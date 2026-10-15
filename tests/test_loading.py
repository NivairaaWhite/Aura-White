import numpy as np
import pytest
import torch

from aura_white import AuraWhite
from aura_white.config import AuraConfig
from aura_white.model import AuraNet
from aura_white.weights import convert

CFG = AuraConfig.tiny()


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(3)
    return AuraNet(CFG).eval()


def test_pickle_checkpoint_roundtrip(tmp_path, reference):
    p = tmp_path / "model.ckpt"
    torch.save(reference.state_dict(), p)  # what an official-style .ckpt looks like
    eng = AuraWhite.from_pretrained(str(p), device="cpu", cfg=CFG)
    x = torch.rand(1, 3, 64, 64)
    assert torch.allclose(eng.net.encode(x), reference.encode(x), atol=1e-6)


@pytest.mark.parametrize("dtype,tol", [("fp32", 1e-6), ("fp16", 5e-2)])
def test_converted_safetensors_roundtrip(tmp_path, reference, dtype, tol):
    pytest.importorskip("safetensors")
    src = tmp_path / "model.ckpt"
    torch.save(reference.state_dict(), src)
    out = convert(src, tmp_path / f"w_{dtype}.safetensors", dtype)
    assert out.suffix == ".safetensors"
    if dtype == "fp16":
        assert out.stat().st_size < 0.6 * src.stat().st_size
    eng = AuraWhite.from_pretrained(str(out), device="cpu", cfg=CFG)
    x = torch.rand(1, 3, 64, 64)
    a, b = eng.net.encode(x), reference.encode(x)
    assert (a - b).abs().max() <= tol * b.abs().max()


def test_wrong_architecture_gives_readable_error(tmp_path, reference):
    other = AuraConfig.tiny()
    other.bb_layers = 3
    p = tmp_path / "w.ckpt"
    torch.save(AuraNet(other).state_dict(), p)
    with pytest.raises(RuntimeError, match="architecture"):
        AuraWhite.from_pretrained(str(p), device="cpu", cfg=CFG)


def test_fp16_precision_request_on_cpu_falls_back_to_fp32(tmp_path, reference):
    p = tmp_path / "m.ckpt"
    torch.save(reference.state_dict(), p)
    eng = AuraWhite.from_pretrained(str(p), device="cpu", precision="fp16", cfg=CFG)
    assert eng.dtype == torch.float32


def test_bf16_runs_on_cpu(tmp_path, reference):
    p = tmp_path / "m.ckpt"
    torch.save(reference.state_dict(), p)
    eng = AuraWhite.from_pretrained(str(p), device="cpu", precision="bf16", cfg=CFG)
    x = torch.rand(1, 3, 64, 64)
    a, b = eng.net.encode(x), reference.encode(x)
    assert a.dtype == torch.float32 and (a - b).abs().max() < 0.2 * b.abs().max()


def test_end_to_end_from_saved_weights(tmp_path, reference):
    from PIL import Image, ImageDraw
    p = tmp_path / "m.ckpt"
    torch.save(reference.state_dict(), p)
    eng = AuraWhite.from_pretrained(str(p), device="cpu", cfg=CFG)
    im = Image.new("RGB", (120, 120), (235, 235, 235))
    ImageDraw.Draw(im).ellipse((30, 20, 90, 100), fill=(30, 90, 200))
    r = eng.generate(im, resolution=48, threshold="auto")
    ref = AuraWhite(reference, torch.device("cpu"), torch.float32).generate(im, resolution=48, threshold="auto")
    assert np.array_equal(r.faces, ref.faces) and np.allclose(r.vertices, ref.vertices, atol=1e-5)
