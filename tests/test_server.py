import io
import json
import struct
import threading
import time
import urllib.error
import urllib.request

import pytest
from PIL import Image, ImageDraw

from aura_white import server as srv
from aura_white.pipeline import AuraWhite


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    app = srv.App(out, lambda progress: AuraWhite.from_random())
    httpd = srv.make_server(app, "127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", out
    httpd.shutdown()


def png():
    im = Image.new("RGB", (160, 160), (235, 235, 235))
    ImageDraw.Draw(im).ellipse((40, 30, 120, 130), fill=(200, 60, 50))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def call(url, data=None, method=None):
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_page_and_health(base):
    url, _ = base
    st, body, h = call(url + "/")
    assert st == 200 and b"Aura White" in body and "text/html" in h["Content-Type"]
    st, body, _ = call(url + "/api/health")
    assert st == 200 and json.loads(body)["ok"] is True


def test_full_job_flow(base):
    url, out = base
    st, body, _ = call(url + "/api/jobs?resolution=64&smooth=1&threshold=auto&formats=glb,obj,ply,stl", png(), "POST")
    assert st == 202
    jid = json.loads(body)["id"]
    for _ in range(300):
        st, body, _ = call(f"{url}/api/jobs/{jid}")
        j = json.loads(body)
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert j["status"] == "done", j
    files = j["result"]["files"]
    assert set(files) >= {"glb", "obj", "ply", "stl", "view", "input"}
    st, glb, h = call(url + files["glb"])
    assert st == 200 and glb[:4] == b"glTF" and "attachment" in h["Content-Disposition"]
    st, view, _ = call(url + files["view"])
    nv, nf = struct.unpack_from("<II", view, 4)
    assert view[:4] == b"AWV1" and nv == j["result"]["stats"]["vertices"] and nf == j["result"]["stats"]["faces"]
    st, inp, _ = call(url + files["input"])
    assert Image.open(io.BytesIO(inp)).size[0] > 0
    assert (out / jid / "mesh.obj").exists()


def test_bad_inputs_and_traversal(base, monkeypatch):
    url, _ = base
    assert call(url + "/api/jobs?resolution=abc", png(), "POST")[0] == 400
    assert call(url + "/api/jobs", b"", "POST")[0] in (400, 411)
    assert call(url + "/api/jobs/zzzzzzzzzzzz")[0] == 404
    for evil in ("/files/../../etc/passwd", "/files/000000000000/../../../etc/passwd", "/files/000000000000/%2e%2e%2fsecret", "/files/abc/mesh.glb"):
        assert call(url + evil)[0] == 404
    monkeypatch.setattr(srv, "MAX_UPLOAD", 10)
    assert call(url + "/api/jobs", png(), "POST")[0] == 413


def test_not_an_image_reports_error_and_server_survives(base):
    url, _ = base
    st, body, _ = call(url + "/api/jobs?resolution=64", b"this is not an image", "POST")
    jid = json.loads(body)["id"]
    for _ in range(100):
        j = json.loads(call(f"{url}/api/jobs/{jid}")[1])
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "error" and j["error"]
    assert call(url + "/api/health")[0] == 200
