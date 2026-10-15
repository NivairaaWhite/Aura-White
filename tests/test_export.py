import json
import struct

import numpy as np
import pytest

from aura_white.mesh import glb_bytes, save_mesh, surface_nets, view_bytes

trimesh = pytest.importorskip("trimesh")


@pytest.fixture(scope="module")
def sphere():
    t = np.linspace(-1, 1, 40, dtype=np.float32)
    x, y, z = np.meshgrid(t, t, t, indexing="ij")
    v, f = surface_nets(0.6 - np.sqrt(x * x + y * y + z * z), 0.0)
    v = v * (2 / 39) - 1
    rng = np.random.RandomState(0)
    c = rng.randint(0, 256, size=(len(v), 3)).astype(np.uint8)
    return v, f, c


@pytest.mark.parametrize("fmt", ["obj", "ply", "stl", "glb"])
def test_roundtrip_with_trimesh(tmp_path, sphere, fmt):
    v, f, c = sphere
    p = save_mesh(tmp_path / f"m.{fmt}", v, f, c)
    m = trimesh.load(str(p), force="mesh", process=False)
    assert len(m.faces) == len(f)
    assert np.allclose(m.bounds, [v.min(0), v.max(0)], atol=1e-4)
    if fmt != "stl":
        assert len(m.vertices) == len(v)
    m2 = trimesh.load(str(p), force="mesh", process=True)
    assert m2.is_watertight and m2.volume > 0


def test_ply_and_obj_keep_vertex_colors(tmp_path, sphere):
    v, f, c = sphere
    for fmt in ("ply", "obj"):
        m = trimesh.load(str(save_mesh(tmp_path / f"m.{fmt}", v, f, c)), force="mesh", process=False)
        got = np.asarray(m.visual.vertex_colors)[:, :3]
        assert np.abs(got.astype(int) - c.astype(int)).max() <= 1


def test_glb_structure_and_linear_colors(sphere):
    v, f, c = sphere
    blob = glb_bytes(v, f, c)
    magic, version, total = struct.unpack_from("<4sII", blob, 0)
    assert magic == b"glTF" and version == 2 and total == len(blob) and len(blob) % 4 == 0
    jlen, jtype = struct.unpack_from("<II", blob, 12)
    assert jtype == 0x4E4F534A
    gltf = json.loads(blob[20:20 + jlen])
    prim = gltf["meshes"][0]["primitives"][0]
    acc = gltf["accessors"]
    assert acc[prim["attributes"]["POSITION"]]["count"] == len(v)
    assert acc[prim["indices"]]["count"] == f.size
    assert acc[prim["attributes"]["COLOR_0"]]["componentType"] == 5126
    blen, btype = struct.unpack_from("<II", blob, 20 + jlen)
    assert btype == 0x004E4942 and gltf["buffers"][0]["byteLength"] == blen
    # decode COLOR_0 ourselves, straight from the binary buffer
    bin_start = 20 + jlen + 8
    view = gltf["bufferViews"][acc[prim["attributes"]["COLOR_0"]]["bufferView"]]
    raw = np.frombuffer(blob, dtype="<f4", count=len(v) * 4, offset=bin_start + view["byteOffset"]).reshape(-1, 4)
    cf = c / 255.0
    lin = np.where(cf <= 0.04045, cf / 12.92, ((cf + 0.055) / 1.055) ** 2.4)
    assert np.abs(raw[:, :3] - lin).max() < 1e-6 and (raw[:, 3] == 1.0).all()

    # and let trimesh (independent implementation) decode the same accessor: it only exposes
    # vertex colors for material-less primitives, so strip the material from a copy of the JSON
    gltf2 = json.loads(json.dumps(gltf))
    gltf2["meshes"][0]["primitives"][0].pop("material")
    gltf2.pop("materials")
    js2 = json.dumps(gltf2, separators=(",", ":")).encode()
    js2 += b" " * ((4 - len(js2) % 4) % 4)
    rest = blob[20 + jlen:]
    blob2 = (struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js2) + len(rest)) + struct.pack("<II", len(js2), 0x4E4F534A) + js2 + rest)
    m = trimesh.load(__import__("io").BytesIO(blob2), file_type="glb", force="mesh", process=False)
    col = np.asarray(m.visual.vertex_colors)[:, :3].astype(float) / 255.0
    assert np.abs(col - lin).max() < 0.01


def test_glb_without_colors(sphere):
    v, f, _ = sphere
    m = trimesh.load(__import__("io").BytesIO(glb_bytes(v, f, None)), file_type="glb", force="mesh", process=False)
    assert len(m.faces) == len(f)


def test_view_blob_layout(sphere):
    v, f, c = sphere
    b = view_bytes(v, f, c)
    assert b[:4] == b"AWV1"
    nv, nf = struct.unpack_from("<II", b, 4)
    assert (nv, nf) == (len(v), len(f))
    assert len(b) == 12 + nv * 12 + nv * 12 + nv * 4 + nf * 12
    assert (12 + 28 * nv) % 4 == 0
