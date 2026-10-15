"""Training data for Aura Native: one `.npz` per asset + a manifest that carries the provenance and licence ledger.

A record holds everything the two models need, so nothing is recomputed per step:
  pts (N,9)  surface points: xyz, normal, rgb          q (Q,3), sdf (Q), qrgb (Q,3)   point queries with exact signed distance
  view_az / view_el (V)  and  imgs (V,S,S,4) uint8     renders (or the user's own picture) the generator is conditioned on
Sources: Aura's procedural factory (always available, exact), any folder of meshes (`ingest_folder`, with per-asset
licence filtering), and teacher outputs that passed the gates (`teach.py`).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as G
from .licenses import Provenance, asset_licence_ok, can_teach
from .render import render, to_uint8, from_uint8

DEFAULT_VIEWS: Tuple[Tuple[float, float], ...] = ((0, 0), (180, 0), (90, 0), (-90, 0), (35, 20), (-35, 20))


@dataclass
class DataSpec:
    n_points: int = 1024
    n_queries: int = 4096
    image_size: int = 64
    views: Tuple[Tuple[float, float], ...] = DEFAULT_VIEWS
    mesh_res: int = 64                      # procedural mesh used only for rendering
    sdf_res: int = 96                       # real-mesh sign grid

    def as_dict(self):
        d = asdict(self)
        d["views"] = [list(v) for v in self.views]
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["views"] = tuple(tuple(v) for v in d["views"])
        return DataSpec(**d)


# ------------------------------------------------------------------------------------------------ records
def _render_views(v, f, colours, spec: DataSpec):
    imgs, az, el = [], [], []
    for a, e in spec.views:
        im, _, _ = render(v, f, colours, a, e, spec.image_size)
        imgs.append(to_uint8(im))
        az.append(a)
        el.append(e)
    return np.stack(imgs), np.asarray(az, np.float32), np.asarray(el, np.float32)


def record_from_asset(asset, spec: DataSpec, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    pts, nrm = asset.sample_surface(spec.n_points, rng)
    rgb = asset.colour(pts)
    P9 = np.concatenate([pts, nrm, rgb], 1)
    q, sdf, qrgb = asset.sample_queries(spec.n_queries, rng, pts)
    v, f, c = asset.mesh(spec.mesh_res)
    imgs, az, el = _render_views(v, f, c, spec)
    return {"pts": P9.astype(np.float16), "has_colour": np.uint8(1), "q": q.astype(np.float16), "sdf": sdf.astype(np.float16),
            "qrgb": (qrgb * 255).astype(np.uint8), "view_az": az, "view_el": el, "imgs": imgs,
            "meta": np.array(json.dumps({"family": asset.family, "seed": int(asset.seed), "source": "own", **{k: (float(x) if isinstance(x, (np.floating, float)) else x) for k, x in asset.meta.items() if isinstance(x, (bool, int, float, np.floating, str))}}))}


def _colour_lookup(cloud: Optional[Tuple[np.ndarray, np.ndarray]], centre, scale):
    if cloud is None:
        return None
    from scipy.spatial import cKDTree

    p = (np.asarray(cloud[0], np.float64) - centre) * scale
    tree = cKDTree(p)
    rgb = np.asarray(cloud[1], np.float32) / 255.0

    def look(x):
        _, i = tree.query(x, k=1, workers=-1)
        return rgb[i]
    return look


def record_from_mesh(verts, faces, cloud, spec: DataSpec, rng: np.random.Generator, source: str = "own", licence: str = "own",
                     picture: Optional[np.ndarray] = None, meta: Optional[Dict] = None) -> Optional[Dict[str, np.ndarray]]:
    """A real / teacher mesh -> a record.  `cloud` = (points, rgb uint8) sampled from the mesh's own texture, or None.
    `picture` (S,S,4 float, prepared by render.prep_picture): condition on THIS picture only (teacher pairs, user art);
    otherwise the mesh is rendered from `spec.views`.  Returns None when the mesh is unusable."""
    from scipy.spatial import cKDTree
    from ..gen3d.meshutil import orient_outward

    try:
        v, f, centre, scale = G.normalize_mesh(verts, faces)
        v, f = orient_outward(v, f)
        pts, nrm = G.surface_samples(v, f, spec.n_points, rng)
    except Exception:
        return None
    look = _colour_lookup(cloud, centre, scale)
    rgb = look(pts) if look else np.full((len(pts), 3), 0.7, np.float32)
    P9 = np.concatenate([pts, nrm, rgb], 1)
    # queries: half uniform (sign from the flood-filled grid), half near the surface (exact |d| from a dense KD-tree, sign from the normal)
    dense, dn = G.surface_samples(v, f, 250_000, rng)
    tree = cKDTree(dense)
    nq = spec.n_queries
    U = rng.uniform(-1.0, 1.0, size=(nq // 2, 3))
    near = pts[rng.integers(0, len(pts), size=nq - nq // 2)] + rng.normal(size=(nq - nq // 2, 3)) * rng.choice([0.006, 0.02, 0.06], size=(nq - nq // 2, 1))
    Q = np.concatenate([U, near])
    d, idx = tree.query(Q, k=1, workers=-1)
    grid = G.mesh_sdf_grid(v, f, spec.sdf_res, seed=int(rng.integers(1 << 30)))
    sg = np.sign(G.sample_grid(grid, Q))
    sn = np.sign(((Q - dense[idx]) * dn[idx]).sum(1))
    nearmask = d < 0.04
    sign = np.where(nearmask & (sn != 0), sn, sg)
    agree = float((sn[nearmask] == sg[nearmask]).mean()) if nearmask.any() else 1.0
    sdf = (d * np.where(sign == 0, 1.0, sign)).astype(np.float32)
    qrgb = (look(Q) if look else np.full((len(Q), 3), 0.7, np.float32))
    vcol = look(v) if look else None
    if picture is not None:
        imgs = to_uint8(picture)[None]
        az, el = np.zeros(1, np.float32), np.zeros(1, np.float32)
    else:
        imgs, az, el = _render_views(v, f, vcol, spec)
    m = {"source": source, "licence": licence, "sign_agreement": agree, **(meta or {})}
    return {"pts": P9.astype(np.float16), "has_colour": np.uint8(cloud is not None), "q": Q.astype(np.float16),
            "sdf": sdf.astype(np.float16), "qrgb": (qrgb * 255).astype(np.uint8), "view_az": az, "view_el": el, "imgs": imgs,
            "meta": np.array(json.dumps(m))}


def write_record(path: Path, rec: Dict[str, np.ndarray]):
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **rec)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ the dataset directory
class DataDir:
    """A dataset folder: `manifest.json` + `items/*.npz`.  Append-only; safe to extend with more sources later."""

    def __init__(self, root, spec: Optional[DataSpec] = None):
        self.root = Path(root)
        (self.root / "items").mkdir(parents=True, exist_ok=True)
        mp = self.root / "manifest.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            self.spec = DataSpec.from_dict(m["spec"])
            self.items: List[Dict] = m["items"]
            self.prov = Provenance.from_dict(m.get("provenance", {}))
        else:
            self.spec = spec or DataSpec()
            self.items, self.prov = [], Provenance()

    def save(self):
        (self.root / "manifest.json").write_text(json.dumps({"version": 1, "spec": self.spec.as_dict(), "items": self.items,
                                                             "provenance": self.prov.as_dict()}, indent=1))

    def add(self, rec: Dict[str, np.ndarray], item_id: str, source: str, licence: str = "own", attribution: str = "",
            extra: Optional[Dict] = None):
        name = f"{len(self.items):07d}_{item_id}.npz"[:120]
        write_record(self.root / "items" / name, rec)
        self.items.append({"file": name, "id": item_id, "source": source, "licence": licence, **(extra or {})})
        self.prov.add(source, 1, None, attribution)

    def __len__(self):
        return len(self.items)


def _proc_worker(args):
    root, spec_d, family, seed, idx = args
    from .procedural import make_asset

    spec = DataSpec.from_dict(spec_d)
    rng = np.random.default_rng(seed * 1000003 + 17)
    a = make_asset(family, seed)
    rec = record_from_asset(a, spec, rng)
    name = f"{idx:07d}_{a.family}{seed}.npz"
    write_record(Path(root) / "items" / name, rec)
    return {"file": name, "id": f"{a.family}{seed}", "source": "own", "licence": "own", "family": a.family, "seed": int(seed)}


def build_procedural(root, n: int, spec: Optional[DataSpec] = None, seed0: int = 0, families: Optional[Sequence[str]] = None,
                     workers: int = 0, say=print) -> DataDir:
    """Append `n` procedural assets (seeds seed0 ..) to the dataset at `root`."""
    from .procedural import FAMILIES

    dd = DataDir(root, spec)
    fams = list(families or FAMILIES)
    start = len(dd)
    jobs = [(str(dd.root), dd.spec.as_dict(), fams[(seed0 + i) % len(fams)], seed0 + i, start + i) for i in range(n)]
    if workers and workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for k, it in enumerate(pool.imap(_proc_worker, jobs, chunksize=4)):
                dd.items.append(it)
                if say and (k + 1) % max(1, n // 5) == 0:
                    say(f"  procedural {k + 1}/{n}")
    else:
        for k, j in enumerate(jobs):
            dd.items.append(_proc_worker(j))
            if say and (k + 1) % max(1, n // 5) == 0:
                say(f"  procedural {k + 1}/{n}")
    dd.prov.add("own", n, can_teach("own"))
    dd.save()
    return dd


MESH_EXT = (".glb", ".gltf", ".obj", ".ply")


def ingest_folder(folder, root, spec: Optional[DataSpec] = None, licences: Optional[Dict[str, str]] = None, commercial: bool = True,
                  source: str = "own", default_licence: str = "own", limit: Optional[int] = None, seed: int = 0, say=print) -> DataDir:
    """Add every mesh under `folder` to the dataset.  `licences`: {filename stem or relative path: licence tag} (Objaverse-style
    tags: cc0 / by / by-sa / by-nc ...); an asset whose tag the mode does not accept is skipped and counted.  Files with no
    entry use `default_licence` (the right choice for your own assets)."""
    from ..asset.meshio import load_mesh_coloured

    dd = DataDir(root, spec)
    rng = np.random.default_rng(seed)
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in MESH_EXT)
    if limit:
        files = files[:limit]
    skipped, bad = {}, 0
    for k, p in enumerate(files):
        tag = (licences or {}).get(str(p.relative_to(folder))) or (licences or {}).get(p.stem) or default_licence
        ok, why = asset_licence_ok(tag, commercial)
        if not ok:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        try:
            v, f, cloud = load_mesh_coloured(p, n_samples=400_000)
            rec = record_from_mesh(v, f, cloud, dd.spec, rng, source=source, licence=tag, meta={"file": p.name})
        except Exception as e:                                                 # a broken file must not stop a 50k-asset run
            rec = None
            say(f"  skipped {p.name}: {type(e).__name__}: {e}")
        if rec is None or json.loads(str(rec["meta"]))["sign_agreement"] < 0.8:
            bad += 1
            continue
        dd.add(rec, p.stem, source, tag, extra={"file": p.name})
        if say and (k + 1) % 25 == 0:
            say(f"  ingested {len(dd)} / {k + 1} files")
    dd.prov.commercial = commercial
    dd.save()
    if skipped:
        say("  licence filter skipped: " + "; ".join(f"{n} x {w}" for w, n in skipped.items()))
    if bad:
        say(f"  {bad} files unusable (no area / leaky sign)")
    return dd


# ------------------------------------------------------------------------------------------------ torch dataset
try:
    import torch
    from torch.utils.data import Dataset
except Exception:                                                               # numpy-only environments can still build data
    torch = None
    Dataset = object


class ShardDataset(Dataset):
    """Reads a DataDir.  split: train / val / all (every `val_every`-th item is validation, deterministically)."""

    def __init__(self, root, split: str = "train", val_every: int = 7, cache: bool = True):
        self.dd = DataDir(root)
        idx = list(range(len(self.dd)))
        if split == "val":
            idx = [i for i in idx if i % val_every == 0]
        elif split == "train":
            idx = [i for i in idx if i % val_every != 0]
        self.idx = idx
        self.cache = {} if cache else None

    @property
    def spec(self) -> DataSpec:
        return self.dd.spec

    def __len__(self):
        return len(self.idx)

    def meta(self, i: int) -> Dict:
        return self.dd.items[self.idx[i]]

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        with np.load(self.dd.root / "items" / self.dd.items[self.idx[i]]["file"]) as z:
            rec = {k: z[k] for k in z.files}
        if self.cache is not None:
            self.cache[i] = rec
        return rec
