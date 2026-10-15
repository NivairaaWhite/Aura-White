import json
import struct

import numpy as np

from aura_white.mesh.textured import glb_textured_bytes


def _parse(blob: bytes) -> dict:
    magic, ver, length = struct.unpack("<III", blob[:12])
    assert magic == 0x46546C67
    clen, ctype = struct.unpack("<II", blob[12:20])
    return json.loads(blob[20:20 + clen])


def _cube():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    albedo = np.full((8, 8, 3), 200, np.uint8)
    return v, f, uv, albedo


def test_orm_texture_uses_glTF_occlusion_and_metallic_roughness_channels():
    v, f, uv, albedo = _cube()
    orm = np.zeros((8, 8, 3), np.uint8)
    orm[..., 0], orm[..., 1], orm[..., 2] = 255, 120, 10
    blob = glb_textured_bytes(v, f, uv, albedo, orm=orm)
    doc = _parse(blob)
    mat = doc["materials"][0]
    assert "occlusionTexture" in mat
    assert mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"] == mat["occlusionTexture"]["index"]
    assert len(doc["images"]) == 2                     # albedo + orm


def test_unlit_sets_the_khr_extension_and_advertises_it():
    v, f, uv, albedo = _cube()
    blob = glb_textured_bytes(v, f, uv, albedo, unlit=True)
    doc = _parse(blob)
    assert doc["materials"][0]["extensions"]["KHR_materials_unlit"] == {}
    assert "KHR_materials_unlit" in doc["extensionsUsed"]


def test_no_orm_no_unlit_keeps_the_old_plain_material():
    v, f, uv, albedo = _cube()
    blob = glb_textured_bytes(v, f, uv, albedo)
    doc = _parse(blob)
    assert "occlusionTexture" not in doc["materials"][0]
    assert "extensions" not in doc["materials"][0]
    assert "extensionsUsed" not in doc
    assert len(doc["images"]) == 1


def test_normal_orm_and_albedo_all_present_get_distinct_image_indices():
    v, f, uv, albedo = _cube()
    normal = np.full((8, 8, 3), (128, 128, 255), np.uint8)
    orm = np.full((8, 8, 3), (255, 100, 0), np.uint8)
    blob = glb_textured_bytes(v, f, uv, albedo, normal_map=normal, orm=orm)
    doc = _parse(blob)
    mat = doc["materials"][0]
    idx = {mat["pbrMetallicRoughness"]["baseColorTexture"]["index"], mat["normalTexture"]["index"],
          mat["occlusionTexture"]["index"]}
    assert len(idx) == 3, "albedo, normal and ORM must each get their own texture/image slot"
    assert len(doc["images"]) == 3
