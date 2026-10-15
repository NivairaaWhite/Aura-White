"""Writers for textured meshes (UV + albedo [+ tangent-space normal map]): binary glTF 2.0 and OBJ/MTL.
NumPy + Pillow only."""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .export import _pad4
from .cleanup import vertex_normals


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def glb_textured_bytes(verts, faces, uv, albedo: np.ndarray, normals=None, normal_map: Optional[np.ndarray] = None,
                       tangents: Optional[np.ndarray] = None, metallic: float = 0.0, roughness: float = 0.7,
                       double_sided: bool = True, name: str = "AuraWhite", orm: Optional[np.ndarray] = None,
                       unlit: bool = False) -> bytes:
    """glTF 2.0 binary with baseColorTexture (+ normalTexture and TANGENT when given).
    `uv` uses the glTF convention (origin top-left, v downwards) - the same as this package's atlases."""
    v = np.ascontiguousarray(verts, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    n = vertex_normals(v, f.astype(np.int64)) if normals is None else np.ascontiguousarray(normals, dtype=np.float32)
    t = np.ascontiguousarray(uv, dtype="<f4")
    chunks, views, accessors = [], [], []
    offset = 0

    def add(data: bytes, target: Optional[int], acc: Optional[dict] = None):
        nonlocal offset
        data = _pad4(data)
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            bv["target"] = target
        views.append(bv)
        chunks.append(data)
        offset += len(data)
        if acc is None:
            return len(views) - 1
        accessors.append(dict(acc, bufferView=len(views) - 1, byteOffset=0))
        return len(accessors) - 1

    attrs = {
        "POSITION": add(v.tobytes(), 34962, {"componentType": 5126, "count": len(v), "type": "VEC3",
                                             "min": v.min(0).tolist(), "max": v.max(0).tolist()}),
        "NORMAL": add(n.astype("<f4").tobytes(), 34962, {"componentType": 5126, "count": len(n), "type": "VEC3"}),
        "TEXCOORD_0": add(t.tobytes(), 34962, {"componentType": 5126, "count": len(t), "type": "VEC2"}),
    }
    if tangents is not None and normal_map is not None:
        tg = np.ascontiguousarray(tangents, dtype="<f4")
        attrs["TANGENT"] = add(tg.tobytes(), 34962, {"componentType": 5126, "count": len(tg), "type": "VEC4"})
    idx = add(f.astype("<u4").ravel().tobytes(), 34963, {"componentType": 5125, "count": int(f.size), "type": "SCALAR"})

    images, textures = [], []
    images.append({"bufferView": add(_png(albedo), None), "mimeType": "image/png", "name": "albedo"})
    textures.append({"source": 0, "sampler": 0})
    material = {"name": "AuraWhiteMaterial", "doubleSided": bool(double_sided),
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0, "texCoord": 0},
                                         "baseColorFactor": [1, 1, 1, 1], "metallicFactor": float(metallic),
                                         "roughnessFactor": float(roughness)}}
    if normal_map is not None:
        images.append({"bufferView": add(_png(normal_map), None), "mimeType": "image/png", "name": "normal"})
        textures.append({"source": len(images) - 1, "sampler": 0})
        material["normalTexture"] = {"index": len(textures) - 1, "texCoord": 0, "scale": 1.0}
    if orm is not None:
        # glTF convention: R=occlusion, G=roughness, B=metallic - the same channel order this pipeline bakes ORM in
        images.append({"bufferView": add(_png(orm), None), "mimeType": "image/png", "name": "orm"})
        textures.append({"source": len(images) - 1, "sampler": 0})
        material["occlusionTexture"] = {"index": len(textures) - 1, "texCoord": 0, "strength": 1.0}
        material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": len(textures) - 1, "texCoord": 0}
    if unlit:
        material.setdefault("extensions", {})["KHR_materials_unlit"] = {}
    extensions_used = ["KHR_materials_unlit"] if unlit else []
    gltf = {
        "asset": {"version": "2.0", "generator": "Aura White 4.0"},
        **({"extensionsUsed": extensions_used} if extensions_used else {}),
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": attrs, "indices": idx, "material": 0, "mode": 4}]}],
        "materials": [material], "textures": textures, "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "buffers": [{"byteLength": offset}], "bufferViews": views, "accessors": accessors,
    }
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_ = b"".join(chunks)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(bin_), 0x004E4942) + bin_)


def write_obj_textured(path, verts, faces, uv, albedo, normals=None, normal_map=None, name: str = "AuraWhite"):
    """OBJ + MTL + PNG textures next to it (v flipped to the OBJ bottom-left convention)."""
    path = Path(path)
    stem = path.stem
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64) + 1
    n = vertex_normals(v.astype(np.float32), f - 1) if normals is None else np.asarray(normals)
    vt = np.asarray(uv, dtype=np.float64).copy()
    vt[:, 1] = 1.0 - vt[:, 1]
    lines = [f"mtllib {stem}.mtl", f"o {name}"]
    lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v]
    lines += [f"vt {a:.6f} {b:.6f}" for a, b in vt]
    lines += [f"vn {a:.5f} {b:.5f} {c:.5f}" for a, b, c in n]
    lines.append(f"usemtl {name}")
    lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in f]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Image.fromarray(albedo).save(path.with_name(f"{stem}_albedo.png"))
    mtl = [f"newmtl {name}", "Ka 1 1 1", "Kd 1 1 1", "Ks 0 0 0", "d 1", "illum 1", f"map_Kd {stem}_albedo.png"]
    if normal_map is not None:
        Image.fromarray(normal_map).save(path.with_name(f"{stem}_normal.png"))
        mtl.append(f"norm {stem}_normal.png")
    path.with_suffix(".mtl").write_text("\n".join(mtl) + "\n", encoding="utf-8")
    return path


def to_up_axis(x: np.ndarray, up: str = "y") -> np.ndarray:
    """Canonical frame (z up, front +x, right +y) -> glTF/Blender-viewer frame (y up, front +z, right +x).
    Works for positions, normals and tangent xyz alike (proper rotation)."""
    x = np.asarray(x)
    if up == "z":
        return x
    out = x.copy()
    out[..., 0], out[..., 1], out[..., 2] = x[..., 1], x[..., 2], x[..., 0]
    return out


def view_bytes_textured(verts, faces, uv, normals=None) -> bytes:
    """Blob for the bundled WebGL viewer: 'AWV2', nV, nF, f32 pos, f32 nrm, f32 uv, u32 idx."""
    v = np.ascontiguousarray(verts, dtype="<f4")
    f = np.ascontiguousarray(faces, dtype="<u4")
    n = vertex_normals(v, f.astype(np.int64)) if normals is None else np.ascontiguousarray(normals, dtype="<f4")
    t = np.ascontiguousarray(uv, dtype="<f4")
    return b"".join([b"AWV2", struct.pack("<II", len(v), len(f)), v.tobytes(), n.astype("<f4").tobytes(),
                     t.tobytes(), f.tobytes()])
