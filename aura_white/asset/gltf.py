"""A small multi-node binary glTF 2.0 writer: several meshes (LODs, collision hulls) sharing materials and textures.

NumPy + Pillow only.  Positions are in metres, y up, +z towards the front of the asset (glTF's own convention).
"""
from __future__ import annotations

import io
import json
import struct
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def write_glb(path, meshes: Sequence[dict], materials: Sequence[dict], images: Dict[str, np.ndarray],
              lod_group: Optional[Sequence[int]] = None, generator: str = "Aura White asset", extras: Optional[dict] = None) -> int:
    """meshes: dicts with name, verts, faces and optional uv, normals, tangents, material (index), extras, hidden (bool).
    materials: dicts with name, base / normal / orm / emissive (keys of `images`), unlit, metallic, roughness,
    double_sided, emissive_factor.  lod_group: mesh indices [LOD0, LOD1 ...] -> MSFT_lod on the first one.
    Returns the number of bytes written."""
    chunks: List[bytes] = []
    views: List[dict] = []
    accessors: List[dict] = []
    offset = 0

    def add(data: bytes, target: Optional[int] = None, acc: Optional[dict] = None):
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

    img_list, tex_list, tex_of = [], [], {}

    def texture(key: str) -> int:
        if key not in tex_of:
            img_list.append({"bufferView": add(_png(images[key])), "mimeType": "image/png", "name": key})
            tex_list.append({"source": len(img_list) - 1, "sampler": 0})
            tex_of[key] = len(tex_list) - 1
        return tex_of[key]

    gm, ext_used = [], set()
    for m in materials:
        mat = {"name": m.get("name", "material"), "doubleSided": bool(m.get("double_sided", False)),
               "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": float(m.get("metallic", 0.0)),
                                        "roughnessFactor": float(m.get("roughness", 0.8))}}
        if m.get("base"):
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": texture(m["base"]), "texCoord": 0}
        if m.get("normal"):
            mat["normalTexture"] = {"index": texture(m["normal"]), "texCoord": 0, "scale": 1.0}
        if m.get("orm"):
            t = texture(m["orm"])
            mat["occlusionTexture"] = {"index": t, "texCoord": 0, "strength": 1.0}
            mat["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": t, "texCoord": 0}
            mat["pbrMetallicRoughness"]["metallicFactor"] = 1.0
            mat["pbrMetallicRoughness"]["roughnessFactor"] = 1.0
        if m.get("emissive"):
            mat["emissiveTexture"] = {"index": texture(m["emissive"]), "texCoord": 0}
            mat["emissiveFactor"] = list(m.get("emissive_factor", [1.0, 1.0, 1.0]))
        if m.get("unlit"):
            mat.setdefault("extensions", {})["KHR_materials_unlit"] = {}
            ext_used.add("KHR_materials_unlit")
        gm.append(mat)

    gmeshes, nodes = [], []
    for me in meshes:
        v = np.ascontiguousarray(me["verts"], np.float32)
        f = np.ascontiguousarray(me["faces"], np.uint32)
        attrs = {"POSITION": add(v.tobytes(), 34962, {"componentType": 5126, "count": len(v), "type": "VEC3",
                                                      "min": v.min(0).tolist(), "max": v.max(0).tolist()})}
        if me.get("normals") is not None:
            n = np.ascontiguousarray(me["normals"], "<f4")
            attrs["NORMAL"] = add(n.tobytes(), 34962, {"componentType": 5126, "count": len(n), "type": "VEC3"})
        if me.get("uv") is not None:
            t = np.ascontiguousarray(me["uv"], "<f4")
            attrs["TEXCOORD_0"] = add(t.tobytes(), 34962, {"componentType": 5126, "count": len(t), "type": "VEC2"})
        if me.get("tangents") is not None:
            tg = np.ascontiguousarray(me["tangents"], "<f4")
            attrs["TANGENT"] = add(tg.tobytes(), 34962, {"componentType": 5126, "count": len(tg), "type": "VEC4"})
        idx = add(f.astype("<u4").ravel().tobytes(), 34963, {"componentType": 5125, "count": int(f.size), "type": "SCALAR"})
        prim = {"attributes": attrs, "indices": idx, "mode": 4}
        if me.get("material") is not None:
            prim["material"] = int(me["material"])
        gmeshes.append({"name": me["name"], "primitives": [prim]})
        node = {"mesh": len(gmeshes) - 1, "name": me["name"]}
        if me.get("extras"):
            node["extras"] = me["extras"]
        nodes.append(node)
    if lod_group and len(lod_group) > 1:
        nodes[lod_group[0]].setdefault("extensions", {})["MSFT_lod"] = {"ids": [int(i) for i in lod_group[1:]]}
        ext_used.add("MSFT_lod")
    roots = [i for i in range(len(nodes)) if not (lod_group and i in lod_group[1:])]
    gltf = {
        "asset": {"version": "2.0", "generator": generator, **({"extras": extras} if extras else {})},
        "scene": 0, "scenes": [{"nodes": roots}], "nodes": nodes, "meshes": gmeshes,
        "buffers": [{"byteLength": offset}], "bufferViews": views, "accessors": accessors,
    }
    if gm:
        gltf["materials"] = gm
    if tex_list:
        gltf["textures"], gltf["images"] = tex_list, img_list
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
    if ext_used:
        gltf["extensionsUsed"] = sorted(ext_used)
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_ = b"".join(chunks)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    blob = (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(bin_), 0x004E4942) + bin_)
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)
