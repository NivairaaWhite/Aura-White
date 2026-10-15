import http.server
import re
import threading

import pytest
import torch

from aura_white import weights as W
from aura_white.config import AuraConfig
from aura_white.model import AuraNet


def to_new_style(k: str) -> str:
    """Inverse of the renames newer transformers versions apply to the ViT layers."""
    k = re.sub(r"\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r".layers.\1.attention.q_proj.", k)
    k = re.sub(r"\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r".layers.\1.attention.k_proj.", k)
    k = re.sub(r"\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r".layers.\1.attention.v_proj.", k)
    k = re.sub(r"\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r".layers.\1.attention.o_proj.", k)
    k = re.sub(r"\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r".layers.\1.mlp.fc1.", k)
    k = re.sub(r"\.encoder\.layer\.(\d+)\.output\.dense\.", r".layers.\1.mlp.fc2.", k)
    k = re.sub(r"\.encoder\.layer\.(\d+)\.(layernorm_before|layernorm_after)\.", r".layers.\1.\2.", k)
    return k


def test_normalize_handles_old_new_and_pooler():
    sd = AuraNet(AuraConfig.tiny()).state_dict()
    want = set(sd)
    new = {to_new_style(k): v for k, v in sd.items()}
    assert set(new) != want  # the inverse really changed something
    new["image_tokenizer.model.pooler.dense.weight"] = torch.zeros(1)
    assert set(W.normalize_state_dict(new)) == want
    old = dict(sd)
    old["image_tokenizer.model.pooler.dense.bias"] = torch.zeros(1)
    assert set(W.normalize_state_dict(old)) == want
    wrapped = {"model." + k: v for k, v in sd.items()}
    assert set(W.normalize_state_dict(wrapped)) == want


def test_find_existing_triposr(tmp_path, monkeypatch):
    snap = tmp_path / "models--stabilityai--TripoSR" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.ckpt").write_bytes(b"x")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert W.find_existing_triposr() == snap / "model.ckpt"


BLOB = bytes(range(256)) * 12000  # ~3 MB


def make_server(fail_first: bool = False):
    state = {"hits": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state["hits"] += 1
            rng = self.headers.get("Range")
            start = int(re.match(r"bytes=(\d+)-", rng).group(1)) if rng else 0
            body = BLOB[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{len(BLOB) - 1}/{len(BLOB)}")
            self.end_headers()
            if fail_first and state["hits"] == 1:
                self.wfile.write(body[: len(body) // 3])  # drop the connection mid-file
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(body)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, state


def test_download_resumes_partial_file(tmp_path):
    httpd, state = make_server()
    try:
        dest = tmp_path / "m.bin"
        (tmp_path / "m.bin.part").write_bytes(BLOB[:1_000_000])
        seen = []
        W.download(f"http://127.0.0.1:{httpd.server_address[1]}/m", dest, lambda m, f: seen.append(f))
        assert dest.read_bytes() == BLOB and not (tmp_path / "m.bin.part").exists()
        assert seen and seen[-1] == 1.0
    finally:
        httpd.shutdown()


def test_download_survives_dropped_connection(tmp_path):
    httpd, state = make_server(fail_first=True)
    try:
        dest = tmp_path / "m.bin"
        W.download(f"http://127.0.0.1:{httpd.server_address[1]}/m", dest)
        assert dest.read_bytes() == BLOB and state["hits"] >= 2
    finally:
        httpd.shutdown()


def test_resolve_weights_prefers_cache_and_validates_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_WHITE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "nohf"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        W.resolve_weights(str(tmp_path / "missing.safetensors"))
    with pytest.raises(FileNotFoundError):
        W.resolve_weights(None, allow_download=False)
    f = W.weights_dir() / W.FP16_NAME
    f.write_bytes(b"x")
    assert W.resolve_weights(None, allow_download=False) == f
