import io
import json
import threading
import time
import urllib.request

import pytest
from synth import FakeGeometry, FakeMultiView, make_artwork, make_object

from aura_white import server as srv
from aura_white.raster import get_raster
from aura_white.studio import AuraStudio, StudioOptions


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    R = get_raster("cpu")
    V, F = make_object(3)

    def studio_factory():
        return AuraStudio(geometry=FakeGeometry(V, F), multiview=FakeMultiView(R, V, F),
                          options=StudioOptions(quality="draft", texture_size=256, target_faces=3000), device="cpu")

    app = srv.App(out, lambda progress: (_ for _ in ()).throw(RuntimeError("fast engine not needed")), studio_factory)
    httpd = srv.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", out, make_artwork(R, V, F, size=512)
    httpd.shutdown()


def post(url, data):
    req = urllib.request.Request(url, data=data, method="POST")
    return json.loads(urllib.request.urlopen(req).read())


def wait(base_url, job, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = json.loads(urllib.request.urlopen(f"{base_url}/api/jobs/{job}").read())
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.5)
    raise TimeoutError


def test_studio_job_produces_textured_files_and_viewer_blob(base):
    url, out, art = base
    buf = io.BytesIO()
    art.save(buf, "PNG")
    jid = post(f"{url}/api/jobs?mode=studio&quality=draft&formats=glb,obj", buf.getvalue())["id"]
    j = wait(url, jid)
    assert j["status"] == "done", j
    res = j["result"]
    assert res["mode"] == "studio" and res["stats"]["faces"] > 100 and res["stats"]["texture_size"] >= 256   # source-locked default: native artwork size
    for k in ("glb", "obj", "albedo", "view", "texture", "report", "preview"):
        assert k in res["files"], k
    blob = urllib.request.urlopen(url + res["files"]["view"]).read()
    assert blob[:4] == b"AWV2"
    png = urllib.request.urlopen(url + res["files"]["albedo"]).read()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    glb = urllib.request.urlopen(url + res["files"]["glb"]).read()
    assert glb[:4] == b"glTF"


def test_bad_quality_is_rejected(base):
    url, _, _ = base
    with pytest.raises(urllib.error.HTTPError) as e:
        post(f"{url}/api/jobs?mode=studio&quality=godlike", b"x" * 10)
    assert e.value.code == 400


def test_studio_mode_without_factory_reports_a_clear_error(tmp_path):
    app = srv.App(tmp_path, lambda p: None, None)
    httpd = srv.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        jid = post(f"{url}/api/jobs?mode=studio", b"not an image")["id"]
        j = wait(url, jid, 30)
        assert j["status"] == "error" and "not enabled" in j["error"]
    finally:
        httpd.shutdown()


def test_index_page_contains_the_textured_viewer():
    from aura_white.viewer import INDEX_HTML

    assert "AWV2" in INDEX_HTML and "Sharp textured" in INDEX_HTML and "uTex" in INDEX_HTML
