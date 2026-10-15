"""Mesh writers with no third-party dependency (numpy + stdlib only): GLB, OBJ, PLY, STL,
plus a compact binary blob for the built-in web viewer."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .cleanup import vertex_normals

FORMATS = ("glb", "obj", "ply", "stl")


def _prep(verts, faces, colors, normals):
    v = np.ascontiguousarray(verts, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    c = None if colors is None else np.ascontiguousarray(colors, dtype=np.uint8)
    n = vertex_normals(v, f.astype(np.int64)) if normals is None else np.ascontiguousarray(normals, dtype=np.float32)
    return v, f, c, n


def _srgb_to_linear(c8: np.ndarray) -> np.ndarray:
    c = c8.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def glb_bytes(verts, faces, colors=None, normals=None, linear_colors: bool = True, name: str = "AuraWhite") -> bytes:
    """Binary glTF 2.0. Vertex colors are sRGB->linear converted because glTF defines COLOR_0 as linear."""
    v, f, c, n = _prep(verts, faces, colors, normals)
    chunks, views, accessors = [], [], []
    offset = 0

    def add(data: bytes, target: int, acc: dict):
        nonlocal offset
        data = _pad4(data)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        acc = dict(acc, bufferView=len(views) - 1, byteOffset=0)
        accessors.append(acc)
        chunks.append(data)
        offset += len(data)
        return len(accessors) - 1

    attrs = {}
    attrs["POSITION"] = add(v.tobytes(), 34962, {"componentType": 5126, "count": len(v), "type": "VEC3",
                                                  "min": v.min(0).tolist(), "max": v.max(0).tolist()})
    attrs["NORMAL"] = add(n.astype("<f4").tobytes(), 34962, {"componentType": 5126, "count": len(n), "type": "VEC3"})
    if c is not None:
        lin = _srgb_to_linear(c) if linear_colors else c.astype(np.float32) / 255.0
        # float32 (not normalised ints): keeps dark-tone precision after the sRGB->linear
        # conversion and is the encoding every glTF reader supports
        rgba = np.concatenate([lin, np.ones((len(lin), 1), np.float32)], axis=1).astype("<f4")
        attrs["COLOR_0"] = add(rgba.tobytes(), 34962, {"componentType": 5126, "count": len(rgba), "type": "VEC4"})
    idx = add(f.astype("<u4").ravel().tobytes(), 34963, {"componentType": 5125, "count": int(f.size), "type": "SCALAR"})

    gltf = {
        "asset": {"version": "2.0", "generator": "Aura White"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": attrs, "indices": idx, "material": 0, "mode": 4}]}],
        "materials": [{"name": "AuraWhiteMaterial", "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.85}}],
        "buffers": [{"byteLength": offset}],
        "bufferViews": views,
        "accessors": accessors,
    }
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_ = b"".join(chunks)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(bin_), 0x004E4942) + bin_)


def _write_obj(path: Path, v, f, c, n):
    with open(path, "w", newline="\n", encoding="ascii") as fh:
        fh.write("# Aura White\n")
        if c is not None:
            np.savetxt(fh, np.concatenate([v, c.astype(np.float32) / 255.0], axis=1), fmt="v %.6f %.6f %.6f %.4f %.4f %.4f")
        else:
            np.savetxt(fh, v, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, n, fmt="vn %.5f %.5f %.5f")
        idx = f.astype(np.int64) + 1
        tri = np.stack([idx[:, 0], idx[:, 0], idx[:, 1], idx[:, 1], idx[:, 2], idx[:, 2]], axis=1)
        np.savetxt(fh, tri, fmt="f %d//%d %d//%d %d//%d")


def _write_ply(path: Path, v, f, c, n):
    vd = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if c is not None:
        vd += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    va = np.empty(len(v), dtype=vd)
    va["x"], va["y"], va["z"] = v[:, 0], v[:, 1], v[:, 2]
    va["nx"], va["ny"], va["nz"] = n[:, 0], n[:, 1], n[:, 2]
    if c is not None:
        va["red"], va["green"], va["blue"] = c[:, 0], c[:, 1], c[:, 2]
    fa = np.empty(len(f), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    fa["n"], fa["i"] = 3, f.astype(np.int32)
    head = ["ply", "format binary_little_endian 1.0", "comment Aura White", f"element vertex {len(v)}",
            "property float x", "property float y", "property float z",
            "property float nx", "property float ny", "property float nz"]
    if c is not None:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    head += [f"element face {len(f)}", "property list uchar int vertex_indices", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(va.tobytes())
        fh.write(fa.tobytes())


def _write_stl(path: Path, v, f):
    p = v[f.astype(np.int64)]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, p
    with open(path, "wb") as fh:
        fh.write(b"Aura White binary STL".ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())


def save_mesh(path, verts, faces, colors=None, normals=None, fmt: str | None = None, linear_glb_colors: bool = True) -> Path:
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt not in FORMATS:
        raise ValueError(f"unsupported format {fmt!r}; choose from {FORMATS}")
    path.parent.mkdir(parents=True, exist_ok=True)
    v, f, c, n = _prep(verts, faces, colors, normals)
    if fmt == "glb":
        path.write_bytes(glb_bytes(v, f, c, n, linear_colors=linear_glb_colors))
    elif fmt == "obj":
        _write_obj(path, v, f, c, n)
    elif fmt == "ply":
        _write_ply(path, v, f, c, n)
    else:
        _write_stl(path, v, f)
    return path


def view_bytes(verts, faces, colors=None, normals=None) -> bytes:
    """Compact blob for the bundled WebGL viewer: 'AWV1', nV, nF, f32 pos, f32 nrm, u8 rgba, u32 idx."""
    v, f, c, n = _prep(verts, faces, colors, normals)
    rgba = np.full((len(v), 4), 255, np.uint8)
    if c is not None:
        rgba[:, :3] = c
    else:
        rgba[:, :3] = 200
    return b"".join([b"AWV1", struct.pack("<II", len(v), len(f)), v.astype("<f4").tobytes(), n.astype("<f4").tobytes(),
                     rgba.tobytes(), f.astype("<u4").tobytes()])
