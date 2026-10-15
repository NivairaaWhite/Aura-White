"""Read a triangle mesh from a file with no required dependency: GLB / glTF, OBJ, and (through ``trimesh`` when it is
installed) PLY / STL / OFF.  Also the small mesh clean-up and decimation helpers the mesh-lock mode needs.

Everything optional is guarded: quadric decimation uses ``fast_simplification`` when present and falls back to a pure
NumPy vertex-clustering decimator otherwise, so the mode keeps working after a Python or package update.
"""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Tuple

import numpy as np

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


# ------------------------------------------------------------------------------------------------ glTF
def _accessor(j, buffers, idx) -> np.ndarray:
    a = j["accessors"][idx]
    n, nc = a["count"], _NC[a["type"]]
    dt = np.dtype(_CT[a["componentType"]])
    if "bufferView" not in a:
        return np.zeros((n, nc), dt)
    bv = j["bufferViews"][a["bufferView"]]
    buf = buffers[bv["buffer"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride") or nc * dt.itemsize
    need = (n - 1) * stride + nc * dt.itemsize
    raw = np.frombuffer(buf, np.uint8, count=need, offset=start)
    arr = np.lib.stride_tricks.as_strided(raw, shape=(n, nc * dt.itemsize), strides=(stride, 1))
    return np.ascontiguousarray(arr).view(dt).reshape(n, nc).copy()


def _node_matrix(node) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], np.float64).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in node:
        M = np.diag(list(node["scale"]) + [1.0]) @ M
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        Rm = np.eye(4)
        Rm[:3, :3] = R
        M = Rm @ M
    if "translation" in node:
        T = np.eye(4)
        T[:3, 3] = node["translation"]
        M = T @ M
    return M


def _decode_image(j, buffers, path: Path, idx: int):
    """Base-colour image `idx` of a glTF as a uint8 array (h, w, 3 or 4), or None when it cannot be decoded."""
    try:
        from PIL import Image
        import io
        im = j["images"][idx]
        if "bufferView" in im:
            bv = j["bufferViews"][im["bufferView"]]
            raw = bytes(buffers[bv["buffer"]][bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]])
        else:
            uri = im["uri"]
            raw = base64.b64decode(uri.split(",", 1)[1]) if uri.startswith("data:") else (path.parent / uri).read_bytes()
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
        return np.asarray(img)
    except Exception:
        return None


def _sample_texture(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup with glTF conventions (v down, repeat wrapping) -> float (n, 3) in 0..255."""
    h, w = tex.shape[:2]
    x = (uv[:, 0] % 1.0) * w - 0.5
    y = (uv[:, 1] % 1.0) * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, x1 = x0 % w, (x0 + 1) % w
    y0, y1 = y0 % h, (y0 + 1) % h
    t = tex[..., :3].astype(np.float32)
    top = t[y0, x0] * (1 - fx) + t[y0, x1] * fx
    bot = t[y1, x0] * (1 - fx) + t[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def read_gltf(path, colour_samples: int = None):
    """(verts, faces), or (verts, faces, (points, rgb) or None) when `colour_samples` is given: the mesh's own colours
    (vertex colours, or its base-colour texture) read at its vertices plus random surface points, about that many in all."""
    path = Path(path)
    data = path.read_bytes()
    bin_chunk = None
    if data[:4] == b"glTF":
        off = 12
        j = None
        while off + 8 <= len(data):
            ln, ct = struct.unpack("<I4s", data[off:off + 8])
            chunk = data[off + 8:off + 8 + ln]
            off += 8 + ln
            if ct == b"JSON":
                j = json.loads(chunk.decode("utf-8"))
            elif ct.startswith(b"BIN"):
                bin_chunk = chunk
        if j is None:
            raise ValueError(f"{path.name}: no JSON chunk")
    else:
        j = json.loads(data.decode("utf-8"))
    used = set(j.get("extensionsUsed", [])) & {"KHR_draco_mesh_compression", "EXT_meshopt_compression"}
    if used:
        raise ValueError(f"{path.name} uses {', '.join(sorted(used))}; export it without compression "
                         "(or convert it to OBJ) and try again")
    buffers = []
    for b in j.get("buffers", []):
        uri = b.get("uri")
        if uri is None:
            buffers.append(bin_chunk or b"")
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            buffers.append((path.parent / uri).read_bytes())
    nodes = j.get("nodes", [])
    scene = j.get("scenes", [{}])[j.get("scene", 0)] if j.get("scenes") else {}
    roots = scene.get("nodes") or ([i for i in range(len(nodes))] if nodes else [])
    verts, faces, off = [], [], 0
    col_pts, col_rgb = [], []
    rng = np.random.default_rng(1)
    tex_cache = {}

    def prim_colour(prim, P, F, M):
        """-> (points, rgb) samples of one primitive, or None."""
        attrs = prim["attributes"]
        rgb_v = None
        uv_v = None
        mat = j.get("materials", [{}])[prim["material"]] if "material" in prim and j.get("materials") else {}
        pbr = mat.get("pbrMetallicRoughness", {})
        factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], np.float32)
        if "COLOR_0" in attrs:
            c = _accessor(j, buffers, attrs["COLOR_0"])[:, :3].astype(np.float32)
            a = j["accessors"][attrs["COLOR_0"]]
            if a["componentType"] == 5121:
                c = c / 255.0
            elif a["componentType"] == 5123:
                c = c / 65535.0
            rgb_v = np.clip(c, 0, 1) * 255.0 * factor
        elif "baseColorTexture" in pbr and "TEXCOORD_0" in attrs:
            ti = pbr["baseColorTexture"]["index"]
            src = j.get("textures", [{}])[ti].get("source")
            if src is not None:
                if src not in tex_cache:
                    tex_cache[src] = _decode_image(j, buffers, path, src)
                tex = tex_cache[src]
                if tex is not None:
                    uv_v = _accessor(j, buffers, attrs["TEXCOORD_0"]).astype(np.float64)
                    rgb_v = _sample_texture(tex, uv_v) * factor
        if rgb_v is None:
            return None
        pts = [P]
        cols = [rgb_v]
        extra = int(max(0, colour_samples * len(F) / max(1, sum(len(x) for x in faces) + len(F)) - len(P))) \
            if False else int(max(0, min(colour_samples - len(P), 2 * len(F))))
        if extra > 0 and len(F):
            tri = P[F]
            area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            pr = area / max(area.sum(), 1e-30)
            fi = rng.choice(len(F), size=extra, p=pr)
            b = rng.dirichlet((1.0, 1.0, 1.0), size=extra)
            pts.append((tri[fi] * b[..., None]).sum(1))
            if uv_v is not None:
                cuv = (uv_v[F[fi]] * b[..., None]).sum(1)
                cols.append(_sample_texture(tex, cuv) * factor)
            else:
                cols.append((rgb_v[F[fi]] * b[..., None]).sum(1))
        return np.concatenate(pts), np.concatenate(cols)

    def visit(i, parent):
        nonlocal off
        node = nodes[i]
        M = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in j["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4 or "POSITION" not in prim["attributes"]:
                    continue
                P = _accessor(j, buffers, prim["attributes"]["POSITION"]).astype(np.float64)
                P = P @ M[:3, :3].T + M[:3, 3]
                if "indices" in prim:
                    F = _accessor(j, buffers, prim["indices"]).reshape(-1).astype(np.int64)
                else:
                    F = np.arange(len(P), dtype=np.int64)
                F = F[: len(F) // 3 * 3].reshape(-1, 3)
                if np.linalg.det(M[:3, :3]) < 0:
                    F = F[:, [0, 2, 1]]
                if colour_samples:
                    cs = prim_colour(prim, P, F, M)
                    if cs is not None:
                        col_pts.append(cs[0])
                        col_rgb.append(cs[1])
                verts.append(P)
                faces.append(F + off)
                off += len(P)
        for c in node.get("children", []):
            visit(c, M)

    for r in roots:
        visit(r, np.eye(4))
    if not verts:
        raise ValueError(f"{path.name}: no triangle meshes found")
    if colour_samples is None:
        return np.concatenate(verts), np.concatenate(faces)
    cs = (np.concatenate(col_pts), np.clip(np.concatenate(col_rgb), 0, 255).astype(np.uint8)) if col_pts else None
    return np.concatenate(verts), np.concatenate(faces), cs


# ------------------------------------------------------------------------------------------------ OBJ
def read_obj(path) -> Tuple[np.ndarray, np.ndarray]:
    v, f = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                v.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    k = int(tok.split("/")[0])
                    idx.append(k - 1 if k > 0 else len(v) + k)
                for i in range(1, len(idx) - 1):
                    f.append((idx[0], idx[i], idx[i + 1]))
    if not v or not f:
        raise ValueError(f"{Path(path).name}: no mesh found")
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


def load_mesh(path) -> Tuple[np.ndarray, np.ndarray]:
    """(verts float64 (N,3), faces int64 (F,3)) from .glb .gltf .obj (built in) or .ply .stl .off (trimesh)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".glb", ".gltf"):
        try:
            return read_gltf(p)
        except (ValueError, KeyError) as e:
            err = e
            try:
                import trimesh
                m = trimesh.load(str(p), force="mesh", process=False)
                return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
            except Exception:
                raise err
    if ext == ".obj":
        return read_obj(p)
    try:
        import trimesh
    except ImportError as e:
        raise ValueError(f"{ext} needs the optional `trimesh` package (pip install trimesh); GLB and OBJ work without it") from e
    m = trimesh.load(str(p), force="mesh", process=False)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def load_mesh_coloured(path, n_samples: int = 1_200_000):
    """(verts, faces, colours) where colours is ``(points (M,3), rgb uint8 (M,3))`` sampled from the mesh's own texture or
    vertex colours, or None when the file carries none.  GLB / glTF built in; PLY / OBJ-with-colours through trimesh."""
    p = Path(path)
    if p.suffix.lower() in (".glb", ".gltf"):
        try:
            return read_gltf(p, colour_samples=n_samples)
        except (ValueError, KeyError):
            pass
    v, f = load_mesh(p)
    try:
        import trimesh
        m = trimesh.load(str(p), force="mesh", process=False)
        vis = m.visual
        if len(m.vertices) == len(v):
            if getattr(vis, "kind", None) == "texture":
                vis = vis.to_color()
            c = getattr(vis, "vertex_colors", None)
            if c is not None and len(c) == len(v) and getattr(vis, "kind", None) == "vertex":
                return v, f, (np.asarray(m.vertices, np.float64), np.asarray(c)[:, :3].astype(np.uint8))
    except Exception:
        pass
    return v, f, None


# ------------------------------------------------------------------------------------------------ clean-up
def weld(v: np.ndarray, f: np.ndarray, tol: float):
    """Merge vertices closer than `tol` (grid snapping), drop degenerate faces and unreferenced vertices."""
    q = np.round(v / max(tol, 1e-12)).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    v2 = v[first]
    f2 = inv.reshape(-1)[f]
    return compact(v2, f2)


def compact(v: np.ndarray, f: np.ndarray):
    f = np.asarray(f, np.int64)
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    area2 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    f = f[area2 > 1e-14]
    used = np.zeros(len(v), bool)
    used[f.reshape(-1)] = True
    remap = np.cumsum(used) - 1
    return v[used], remap[f]


def components(f: np.ndarray, n_verts: int) -> Tuple[int, np.ndarray]:
    """Connected components of the faces (through shared vertices): -> (count, label per face)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    g = sparse.coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n_verts, n_verts))
    n, lab = connected_components(g, directed=False)
    return n, lab[f[:, 0]]


def drop_specks(v: np.ndarray, f: np.ndarray, min_share: float = 2e-4, min_faces: int = 12):
    """Remove tiny disconnected fragments (stray floaters) but keep every part that matters (a sparkle, a ribbon)."""
    n, lab = components(f, len(v))
    if n <= 1:
        return v, f
    cnt = np.bincount(lab, minlength=n)
    keep = (cnt >= max(min_faces, int(min_share * len(f))))
    if not keep.any():
        return v, f
    return compact(v, f[keep[lab]])


# ------------------------------------------------------------------------------------------------ decimation
def _cluster_decimate(v: np.ndarray, f: np.ndarray, target: int):
    lo, hi = v.min(0), v.max(0)
    diag = float(np.linalg.norm(hi - lo))
    a, b = diag / 2000.0, diag / 8.0
    best = None
    for _ in range(14):
        c = 0.5 * (a + b)
        q = np.floor((v - lo) / c).astype(np.int64)
        key = q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791
        uniq, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        rep = np.stack([np.bincount(inv, v[:, k]) / cnt for k in range(3)], 1)
        g = inv[f]
        ok = (g[:, 0] != g[:, 1]) & (g[:, 1] != g[:, 2]) & (g[:, 0] != g[:, 2])
        g = np.sort(g[ok], axis=1)
        g = np.unique(g, axis=0)
        best = (rep, g)
        if len(g) > target:
            a = c
        else:
            b = c
    return compact(*best)


def simplify(v: np.ndarray, f: np.ndarray, target: int, aggressiveness: int = 7):
    """Reduce a triangle mesh to about `target` faces.  Quadric edge collapse (fast_simplification) when it is installed,
    vertex clustering otherwise."""
    target = int(target)
    if target >= len(f):
        return v, f
    try:
        import fast_simplification as fs
        v2, f2 = fs.simplify(np.asarray(v, np.float32), np.asarray(f, np.int32), target_reduction=1.0 - target / len(f),
                             agg=aggressiveness)
        v2, f2 = compact(np.asarray(v2, np.float64), np.asarray(f2, np.int64))
        if len(f2) >= 4:
            return v2, f2
    except ImportError:
        pass
    except Exception:       # a decimator that chokes on a pathological mesh must not stop the build
        pass
    return _cluster_decimate(np.asarray(v, np.float64), np.asarray(f, np.int64), target)


def have_quadric() -> bool:
    try:
        import fast_simplification  # noqa: F401
        return True
    except ImportError:
        return False
