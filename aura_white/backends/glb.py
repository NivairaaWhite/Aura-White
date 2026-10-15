"""Minimal, dependency-free GLB / glTF-binary mesh reader (NumPy only).

Reads every triangle primitive of every mesh node, applies node transforms (matrix or TRS, nested), and returns one
merged (verts, faces).  Enough to consume what Pixal3D / TRELLIS.2 / Hunyuan3D export; textures are not read (Aura
White re-textures from the artwork).  Sparse accessors and Draco / meshopt compression are rejected with a clear error."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

_CTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _accessor(doc: dict, bins: List[bytes], idx: int) -> np.ndarray:
    acc = doc["accessors"][idx]
    if "sparse" in acc:
        raise ValueError("sparse glTF accessors are not supported")
    dt = np.dtype(_CTYPE[acc["componentType"]])
    n = _NCOMP[acc["type"]]
    count = acc["count"]
    if "bufferView" not in acc:
        return np.zeros((count, n), dt)
    bv = doc["bufferViews"][acc["bufferView"]]
    buf = bins[bv.get("buffer", 0)]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or dt.itemsize * n
    if stride == dt.itemsize * n:
        a = np.frombuffer(buf, dtype=dt, count=count * n, offset=off).reshape(count, n)
    else:
        raw = np.frombuffer(buf, dtype=np.uint8, count=stride * (count - 1) + dt.itemsize * n, offset=off)
        a = np.lib.stride_tricks.as_strided(raw, shape=(count, dt.itemsize * n), strides=(stride, 1))
        a = np.ascontiguousarray(a).view(dt).reshape(count, n)
    if acc.get("normalized") and dt.kind in "iu":
        a = a.astype(np.float32) / np.iinfo(dt).max
    return a


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in node:
        m = np.diag(list(node["scale"]) + [1.0]) @ m
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        rm = np.eye(4)
        rm[:3, :3] = r
        m = rm @ m
    if "translation" in node:
        t = np.eye(4)
        t[:3, 3] = node["translation"]
        m = t @ m
    return m


def read_glb(path: Union[str, Path], want_uv: bool = False):
    """Returns (verts (V,3) float32, faces (F,3) int64) in the file's own frame (glTF: y up, front +z).
    With `want_uv=True` also returns `uv` ((V,2) float32, or None if the file has no TEXCOORD_0) as a third value."""
    data = Path(path).read_bytes()
    doc, bins = None, []
    if data[:4] == b"glTF":
        _magic, _ver, _length = struct.unpack("<III", data[:12])
        off = 12
        while off + 8 <= len(data):
            clen, ctype = struct.unpack("<II", data[off:off + 8])
            chunk = data[off + 8:off + 8 + clen]
            if ctype == 0x4E4F534A:
                doc = json.loads(chunk.decode("utf-8"))
            elif ctype == 0x004E4942:
                bins.append(chunk)
            off += 8 + clen
    else:
        doc = json.loads(data.decode("utf-8"))
        base = Path(path).parent
        for b in doc.get("buffers", []):
            bins.append((base / b["uri"]).read_bytes())
    if doc is None:
        raise ValueError("not a glTF file")
    for ext in doc.get("extensionsRequired", []):
        if ext in ("KHR_draco_mesh_compression", "EXT_meshopt_compression"):
            raise ValueError(f"compressed glTF ({ext}) is not supported; export without compression")
    scene = doc.get("scenes", [{}])[doc.get("scene", 0)] if doc.get("scenes") else {}
    roots = scene.get("nodes", list(range(len(doc.get("nodes", [])))))
    verts: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    uvs: List[Optional[np.ndarray]] = []
    base = 0

    def visit(i: int, parent: np.ndarray):
        nonlocal base
        node = doc["nodes"][i]
        m = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                pos = _accessor(doc, bins, prim["attributes"]["POSITION"]).astype(np.float64)
                if "indices" in prim:
                    idx = _accessor(doc, bins, prim["indices"]).astype(np.int64).ravel()
                else:
                    idx = np.arange(len(pos), dtype=np.int64)
                idx = idx[: len(idx) // 3 * 3].reshape(-1, 3)
                verts.append((pos @ m[:3, :3].T + m[:3, 3]).astype(np.float32))
                faces.append(idx + base)
                if want_uv:
                    tc = prim["attributes"].get("TEXCOORD_0")
                    uvs.append(_accessor(doc, bins, tc).astype(np.float32) if tc is not None else
                              np.full((len(pos), 2), np.nan, np.float32))
                base += len(pos)
        for c in node.get("children", []):
            visit(c, m)

    for r in roots:
        visit(r, np.eye(4))
    if not verts:
        raise ValueError("the GLB contains no triangle meshes")
    v, f = np.concatenate(verts, 0), np.concatenate(faces, 0)
    if not want_uv:
        return v, f
    uv_all = np.concatenate(uvs, 0) if uvs else None
    if uv_all is not None and np.isnan(uv_all).all():
        uv_all = None
    return v, f, uv_all


def write_glb_mesh(path: Union[str, Path], verts: np.ndarray, faces: np.ndarray, uv: Optional[np.ndarray] = None) -> None:
    """Minimal GLB writer used by tests, the runner scripts' fallbacks, and the Blender tool bridge.
    `uv` (V,2), glTF convention (v downwards), is written as TEXCOORD_0 when given so Blender / other importers
    pick up Aura White's own UV layout instead of inventing a new one."""
    v = np.asarray(verts, np.float32)
    f = np.asarray(faces, np.uint32)
    pad = lambda b: b + b"\x00" * ((4 - len(b) % 4) % 4)
    parts = [pad(v.tobytes()), pad(f.tobytes())]
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(v.tobytes()), "target": 34962},
             {"buffer": 0, "byteOffset": len(parts[0]), "byteLength": len(f.tobytes()), "target": 34963}]
    accessors = [{"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3",
                 "min": v.min(0).tolist(), "max": v.max(0).tolist()},
                {"bufferView": 1, "componentType": 5125, "count": int(f.size), "type": "SCALAR"}]
    attrs = {"POSITION": 0}
    if uv is not None:
        t = np.asarray(uv, np.float32)
        off = sum(len(p) for p in parts)
        parts.append(pad(t.tobytes()))
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(t.tobytes()), "target": 34962})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5126, "count": len(t), "type": "VEC2"})
        attrs["TEXCOORD_0"] = len(accessors) - 1
    blob = b"".join(parts)
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": attrs, "indices": 1, "mode": 4}]}],
           "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors}
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
                           + struct.pack("<II", len(blob), 0x004E4942) + blob)
