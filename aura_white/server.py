"""Local web app: `aura-white serve`. Standard library only (no Gradio/FastAPI/Flask), so a
framework release can't break it. Binds to 127.0.0.1 unless told otherwise."""
from __future__ import annotations

import json
import queue
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

from .mesh.export import FORMATS, view_bytes
from .viewer import INDEX_HTML

MAX_UPLOAD = 40 * 1024 * 1024
_ID = re.compile(r"^[0-9a-f]{12}$")
_FILE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_TYPES = {".glb": "model/gltf-binary", ".obj": "text/plain", ".ply": "application/octet-stream",
          ".stl": "model/stl", ".png": "image/png", ".bin": "application/octet-stream", ".json": "application/json",
          ".mtl": "text/plain"}
_QUALITY = ("draft", "standard", "high", "ultra")


class Job:
    def __init__(self, data: bytes, opts: dict):
        self.id = uuid.uuid4().hex[:12]
        self.data, self.opts = data, opts
        self.status, self.stage, self.progress = "queued", "queued", 0.0
        self.result: Optional[dict] = None
        self.error: Optional[str] = None

    def public(self) -> dict:
        d = {"id": self.id, "status": self.status, "stage": self.stage, "progress": round(self.progress, 3)}
        if self.result is not None:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d


class App:
    def __init__(self, out_dir: Path, engine_factory: Callable[[Callable], object],
                 studio_factory: Optional[Callable[[], object]] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._factory = engine_factory
        self._studio_factory = studio_factory
        self.engine = None
        self.studio = None
        self.jobs: Dict[str, Job] = {}
        self.q: "queue.Queue[Job]" = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, data: bytes, opts: dict) -> Job:
        job = Job(data, opts)
        self.jobs[job.id] = job
        if len(self.jobs) > 60:  # forget the oldest finished jobs (files stay on disk)
            for k in [k for k, j in self.jobs.items() if j.status in ("done", "error")][:20]:
                self.jobs.pop(k, None)
        self.q.put(job)
        return job

    def _worker(self):
        while True:
            job = self.q.get()
            try:
                self._run(job)
            except Exception as e:  # never let one bad image kill the server
                job.status, job.error = "error", f"{type(e).__name__}: {e}"

    def _run_studio(self, job: Job):
        """Sharp textured pipeline (aura_white.studio): albedo + normal map, GLB/OBJ."""
        from .mesh.textured import view_bytes_textured

        if self._studio_factory is None:
            raise RuntimeError("the textured pipeline is not enabled in this server")
        if self.studio is None:
            self.studio = self._studio_factory()
        o = job.opts

        def prog(stage, frac, msg=""):
            job.stage, job.progress = (f"{stage}: {msg}" if msg else stage), min(0.98, max(0.0, frac))

        res = self.studio.generate(job.data, progress=prog, quality=o["quality"])
        d = self.out_dir / job.id
        d.mkdir(parents=True, exist_ok=True)
        saved = res.save_all(d, "asset", tuple(f for f in o["formats"] if f in ("glb", "obj")) or ("glb",), up="y")
        from .mesh.textured import to_up_axis

        (d / "view.bin").write_bytes(view_bytes_textured(to_up_axis(res.verts, "y"), res.faces, res.uv,
                                                         to_up_axis(res.normals, "y")))
        files = {k: f"/files/{job.id}/{p.name}" for k, p in saved.items()}
        files["view"] = f"/files/{job.id}/view.bin"
        files["texture"] = files["albedo"]
        r = res.report
        job.result = {"mode": "studio", "files": files,
                      "stats": {"vertices": r["vertices_final"], "faces": r["faces_final"], "warnings": r["notes"],
                                "texture_size": r["texture_size"], "geometry": r["geometry"],
                                "side_views": r["side_views"], "iou": r["reference_iou"]},
                      "timings": {"total": r["timings"]["total"]}}
        job.status, job.stage, job.progress = "done", "done", 1.0

    def _run(self, job: Job):
        job.status = "running"
        if job.opts.get("mode") == "studio":
            return self._run_studio(job)

        def prog(msg, frac, lo=0.0, hi=1.0):
            job.stage, job.progress = msg, lo + (hi - lo) * frac

        if self.engine is None:
            self.engine = self._factory(lambda m, f: prog(m, f, 0.0, 0.15))
        o = job.opts
        res = self.engine.generate(job.data, resolution=o["resolution"], smooth=o["smooth"], remove_bg=o["remove_bg"],
                                   threshold=o["threshold"],
                                   progress=lambda m, f: prog(m, f, 0.15, 1.0))
        d = self.out_dir / job.id
        d.mkdir(parents=True, exist_ok=True)
        res.prepared.preview.save(d / "input.png")
        files = {f: f"/files/{job.id}/mesh.{f}" for f in res.save_all(d, "mesh", o["formats"], up="y")}
        v = res.oriented("y")
        (d / "view.bin").write_bytes(view_bytes(v, res.faces, res.colors))
        files["view"] = f"/files/{job.id}/view.bin"
        files["input"] = f"/files/{job.id}/input.png"
        job.result = {"files": files, "stats": res.stats, "timings": {k: round(x, 2) for k, x in res.timings.items()}}
        job.status, job.stage, job.progress = "done", "done", 1.0


def _handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "AuraWhite"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: Optional[dict] = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/health":
                eng = app.engine
                return self._json(200, {"ok": True, "model_loaded": eng is not None, "info": eng.info() if eng else None})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})", path)
            if m:
                j = app.jobs.get(m.group(1))
                return self._json(200, j.public()) if j else self._json(404, {"error": "unknown job"})
            m = re.fullmatch(r"/files/([0-9a-f]{12})/([A-Za-z0-9_.-]{1,64})", path)
            if m and _ID.match(m.group(1)) and _FILE.match(m.group(2)):
                f = (app.out_dir / m.group(1) / m.group(2)).resolve()
                if app.out_dir.resolve() in f.parents and f.is_file():
                    ctype = _TYPES.get(f.suffix.lower(), "application/octet-stream")
                    extra = {"Content-Disposition": f'attachment; filename="{f.name}"'} if f.suffix.lower() in (".glb", ".obj", ".ply", ".stl") else None
                    return self._send(200, f.read_bytes(), ctype, extra)
            self._json(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/jobs":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return self._json(411, {"error": "empty upload"})
            if n > MAX_UPLOAD:
                return self._json(413, {"error": "image too large (40 MB max)"})
            data = self.rfile.read(n)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                fm = [x for x in q.get("formats", "glb").split(",") if x in FORMATS] or ["glb"]
                quality = q.get("quality", "standard")
                if quality not in _QUALITY:
                    raise ValueError("quality")
                opts = {"mode": "studio" if q.get("mode") == "studio" else "fast", "quality": quality,
                        "resolution": max(32, min(512, int(q.get("resolution", 256)))),
                        "smooth": max(0, min(10, int(q.get("smooth", 2)))),
                        "remove_bg": q.get("remove_bg", "1") not in ("0", "false"), "formats": tuple(fm),
                        "threshold": "auto" if q.get("threshold") == "auto" else str(max(0.5, min(1000.0, float(q.get("threshold", 25)))))}
            except ValueError:
                return self._json(400, {"error": "bad parameter"})
            self._json(202, {"id": app.submit(data, opts).id})

    return H


def make_server(app: App, host: str = "127.0.0.1", port: int = 7860) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler(app))


def serve(host: str, port: int, out_dir, engine_factory, preload: bool = False, open_browser: bool = True,
          studio_factory: Optional[Callable[[], object]] = None):
    app = App(Path(out_dir), engine_factory, studio_factory)
    if preload:
        app.engine = engine_factory(lambda m, f: print(f"  {m}", flush=True))
    srv = make_server(app, host, port)
    url = f"http://{host}:{port}/"
    print(f"Aura White is running at {url}   (Ctrl+C to stop)")
    print(f"Files are saved in: {Path(out_dir).resolve()}")
    if open_browser:
        try:
            import webbrowser

            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        srv.server_close()
