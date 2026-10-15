"""Mesh topology repair - NumPy/SciPy only, vectorised, no trimesh / open3d / pymeshlab.

Neural meshes and marching-cubes output arrive with duplicate and degenerate faces, edges shared by three or more
faces, "bow-tie" vertices, inconsistent winding, pin-holes and floating dust.  `repair_mesh` fixes all of that:

  1. drop degenerate (zero-area / repeated-index) and duplicate faces, weld coincident vertices
  2. make the mesh 2-manifold: every edge keeps at most two faces (the pair with the most consistent winding and most
     similar normals is kept together); every vertex is split into one copy per connected fan of faces around it.
     This is done with a connected-components pass over face *corners*, so it is exact and O(F log F)
  3. orient every connected component consistently (BFS over manifold edges), outward-facing for closed shells
  4. fill boundary loops by ear clipping in the loop's best-fit plane (fan + relaxation for very long loops)
  5. remove floating islands (by face count and by relative size)

Every step is reported in the returned dict so a bad input is visible rather than silently "fixed".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RepairOptions:
    weld_tol: float = 1e-6              # relative to the bounding-box diagonal
    min_area: float = 1e-14             # relative to diagonal^2
    make_manifold: bool = True
    orient: bool = True
    fill_holes: bool = True
    max_hole_edges: int = 400
    remove_islands: bool = True
    min_island_faces: int = 24
    min_island_ratio: float = 0.0       # drop shells smaller than this share of the largest shell's faces


# --------------------------------------------------------------------------------------------- basic cleanup
def _compact(v: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    used = np.zeros(len(v), bool)
    used[f.ravel()] = True
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(int(used.sum()))
    return v[used], remap[f]


def face_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def remove_degenerate(v: np.ndarray, f: np.ndarray, min_area: float) -> Tuple[np.ndarray, np.ndarray, int]:
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    keep &= face_areas(v, f) > min_area
    n_deg = int((~keep).sum())
    f = f[keep]
    # duplicate faces (same vertex set)
    key = np.sort(f, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    dup = len(f) - len(first)
    f = f[np.sort(first)]
    v, f = _compact(v, f)
    return v, f, n_deg + dup


def weld(v: np.ndarray, f: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray, int]:
    if tol <= 0 or len(v) == 0:
        return v, f, 0
    q = np.round(v / tol).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    nv = v[first]
    return nv, inv[f], int(len(v) - len(nv))


# --------------------------------------------------------------------------------------------- manifold
def _edge_table(f: np.ndarray):
    """Half-edges: arrays (a, b, face, slot) for the three edges of every face (slot = index of vertex a in face)."""
    F = len(f)
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    face = np.concatenate([np.arange(F)] * 3)
    slot = np.repeat(np.arange(3), F)
    return a, b, face, slot


def _face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-30)


def make_manifold(v: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Return an edge-manifold, vertex-manifold mesh (vertices duplicated where required)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    F = len(f)
    a, b, face, slot = _edge_table(f)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    nV = int(len(v))
    key = lo * nV + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    counts = np.diff(np.r_[starts, len(ks)])
    nrm = _face_normals(v, f)

    pair_a: List[np.ndarray] = []
    pair_b: List[np.ndarray] = []
    extra_corners: List[int] = []
    # ordinary edges (exactly two faces)
    two = starts[counts == 2]
    ha, hb = order[two], order[two + 1]
    pair_a.append(ha)
    pair_b.append(hb)
    # over-shared edges: keep the best pair together
    n_nm_edges = int((counts > 2).sum())
    for s, c in zip(starts[counts > 2], counts[counts > 2]):
        hs = order[s:s + c]
        best, best_score = None, -9.0
        for i in range(c):
            for j in range(i + 1, c):
                # consistent winding = opposite directions along the shared edge
                opposite = 1.0 if a[hs[i]] == b[hs[j]] else 0.0
                score = 2.0 * opposite + float(np.dot(nrm[face[hs[i]]], nrm[face[hs[j]]]) * (1.0 if opposite else -1.0))
                if score > best_score:
                    best_score, best = score, (hs[i], hs[j])
        pair_a.append(np.array([best[0]]))
        pair_b.append(np.array([best[1]]))
        for h in hs:                                  # faces that lose the edge get private corners (they are cut off)
            if h != best[0] and h != best[1]:
                extra_corners += [int(face[h]) * 3 + int(slot[h]), int(face[h]) * 3 + (int(slot[h]) + 1) % 3]
    ha = np.concatenate(pair_a)
    hb = np.concatenate(pair_b)

    # corner nodes: corner id = face * 3 + slot  (slot = position of the vertex in the face)
    def corner(h_face, h_slot):
        return h_face * 3 + h_slot

    fa, sa = face[ha], slot[ha]
    fb, sb = face[hb], slot[hb]
    # half-edge (a->b) in face fa touches corners slot sa (vertex a) and (sa+1)%3 (vertex b)
    ca_a, ca_b = corner(fa, sa), corner(fa, (sa + 1) % 3)
    cb_a, cb_b = corner(fb, sb), corner(fb, (sb + 1) % 3)
    va_a = f[fa, sa]
    # match vertex ids across the two faces (either same or swapped depending on winding)
    same = va_a == f[fb, sb]
    left = np.concatenate([ca_a, ca_b])
    right = np.concatenate([np.where(same, cb_a, cb_b), np.where(same, cb_b, cb_a)])
    G = sparse.coo_matrix((np.ones(len(left)), (left, right)), shape=(3 * F, 3 * F))
    ncomp, comp = connected_components(G, directed=False)
    if extra_corners:
        ec = np.unique(np.asarray(extra_corners, np.int64))
        comp = comp.copy()
        comp[ec] = ncomp + np.arange(len(ec))
        _u, comp = np.unique(comp, return_inverse=True)         # drop ids that lost all their corners
        ncomp = int(len(_u))
    corner_v = np.empty(3 * F, np.int64)
    corner_v[np.arange(F) * 3 + 0] = f[:, 0]
    corner_v[np.arange(F) * 3 + 1] = f[:, 1]
    corner_v[np.arange(F) * 3 + 2] = f[:, 2]
    # a component never spans two vertices, so component -> vertex is well defined
    comp_vertex = np.empty(ncomp, np.int64)
    comp_vertex[comp] = corner_v
    new_v = v[comp_vertex]
    new_f = comp.reshape(F, 3)
    n_split = int(ncomp - nV)
    return new_v, new_f, {"nonmanifold_edges": n_nm_edges, "vertices_split": max(n_split, 0)}


# --------------------------------------------------------------------------------------------- orientation
def _components_faces(f: np.ndarray) -> Tuple[int, np.ndarray]:
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    F = len(f)
    a, b, face, _ = _edge_table(f)
    nV = int(f.max()) + 1 if F else 0
    key = np.minimum(a, b) * nV + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    i0 = order[:-1][same]
    i1 = order[1:][same]
    G = sparse.coo_matrix((np.ones(len(i0)), (face[i0], face[i1])), shape=(F, F))
    return connected_components(G, directed=False)


def orient_consistently(v: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, int]:
    """Flip faces so that neighbours agree on winding; closed shells are made outward-facing.  Meshes whose
    neighbours already agree (the usual case) take a fully vectorised path."""
    from collections import deque

    F = len(f)
    if F == 0:
        return f, 0
    a, b, face, _ = _edge_table(f)
    nV = int(f.max()) + 1
    key = np.minimum(a, b) * nV + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = np.flatnonzero(ks[1:] == ks[:-1])
    h0, h1 = order[same], order[same + 1]
    agree = a[h0] == b[h1]
    vv = v.astype(np.float64)
    g = f.copy()
    flipped_total = 0
    ncomp, lab = _components_faces(f)
    if not agree.all():
        adj_f0, adj_f1 = face[h0], face[h1]
        nbr: List[List[Tuple[int, bool]]] = [[] for _ in range(F)]
        for x, y, ag in zip(adj_f0.tolist(), adj_f1.tolist(), agree.tolist()):
            nbr[x].append((y, ag))
            nbr[y].append((x, ag))
        flip = np.zeros(F, bool)
        seen = np.zeros(F, bool)
        for s in range(F):
            if seen[s]:
                continue
            seen[s] = True
            q = deque([s])
            while q:
                x = q.popleft()
                for y, ag in nbr[x]:
                    if seen[y]:
                        continue
                    flip[y] = flip[x] if ag else (not flip[x])
                    seen[y] = True
                    q.append(y)
        g[flip] = g[flip][:, [0, 2, 1]]
        flipped_total = int(flip.sum())
    vol_face = np.einsum("ij,ij->i", vv[g[:, 0]], np.cross(vv[g[:, 1]], vv[g[:, 2]])) / 6.0
    vol = np.bincount(lab, weights=vol_face, minlength=ncomp)
    neg = (vol < 0)[lab]
    g[neg] = g[neg][:, [0, 2, 1]]
    return g, flipped_total + int(neg.sum())


# --------------------------------------------------------------------------------------------- holes
def boundary_loops(f: np.ndarray) -> List[np.ndarray]:
    """Ordered loops of boundary half-edges (each loop follows the existing face winding).

    Tracks *visited directed edges*, not visited vertices: a "pinch" vertex where two different boundary loops
    touch (common after ear-clipping a concave or self-touching gap) has more than one outgoing boundary
    half-edge, and marking the vertex itself as done - as a simpler implementation would - silently drops
    whichever loop is traced second, leaving a few boundary edges that no later pass will ever revisit."""
    a, b, _, _ = _edge_table(f)
    nV = int(f.max()) + 1 if len(f) else 0
    key = np.minimum(a, b) * nV + np.maximum(a, b)
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bd = cnt[inv] == 1
    ba, bb = a[bd], b[bd]
    nxt: Dict[int, List[int]] = {}
    for x, y in zip(ba.tolist(), bb.tolist()):
        nxt.setdefault(x, []).append(y)
    used_edges = set()
    loops: List[np.ndarray] = []
    for start in list(nxt.keys()):
        for first_y in list(nxt.get(start, ())):
            if (start, first_y) in used_edges:
                continue
            loop = [start]
            used_edges.add((start, first_y))
            cur = first_y
            for _step in range(nV + 4):
                if cur == start:
                    break
                loop.append(cur)
                cand = [y for y in nxt.get(cur, ()) if (cur, y) not in used_edges]
                if not cand:
                    break
                used_edges.add((cur, cand[0]))
                cur = cand[0]
            if cur == start and len(loop) >= 3:
                loops.append(np.asarray(loop, np.int64))
    return loops


def _ear_clip(pts: np.ndarray) -> List[Tuple[int, int, int]]:
    """Ear clipping of a simple polygon given as (n,3) points; returns index triples (CCW in the polygon's plane)."""
    n = len(pts)
    c = pts.mean(0)
    # Newell normal
    nrm = np.zeros(3)
    for i in range(n):
        p, q = pts[i] - c, pts[(i + 1) % n] - c
        nrm += np.cross(p, q)
    ln = np.linalg.norm(nrm)
    if ln < 1e-20:
        return [(0, k, k + 1) for k in range(1, n - 1)]
    nrm /= ln
    ax = np.cross(nrm, np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0]))
    ax /= np.linalg.norm(ax)
    ay = np.cross(nrm, ax)
    P = np.stack([(pts - c) @ ax, (pts - c) @ ay], 1)
    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []

    def cross2(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        m = len(idx)
        best_i, best_q = -1, -1.0
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            if cross2(P[i0], P[i1], P[i2]) <= 1e-18:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                d1, d2, d3 = cross2(P[i0], P[i1], P[j]), cross2(P[i1], P[i2], P[j]), cross2(P[i2], P[i0], P[j])
                if d1 >= -1e-18 and d2 >= -1e-18 and d3 >= -1e-18:
                    ok = False
                    break
            if not ok:
                continue
            e = np.array([np.linalg.norm(P[i1] - P[i0]), np.linalg.norm(P[i2] - P[i1]), np.linalg.norm(P[i0] - P[i2])])
            q = 4.0 * 1.7320508 * abs(cross2(P[i0], P[i1], P[i2])) * 0.5 / max(float((e ** 2).sum()), 1e-30)
            if q > best_q:
                best_q, best_i = q, k
        if best_i < 0:
            best_i = 0                                            # non-simple polygon: clip anyway
        i0, i1, i2 = idx[best_i - 1], idx[best_i], idx[(best_i + 1) % len(idx)]
        tris.append((i0, i1, i2))
        idx.pop(best_i)
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def _split_at_repeats(loop: np.ndarray) -> List[np.ndarray]:
    """A traced boundary loop can revisit the same vertex (a "pinch": two loops sharing one point, or a
    figure-eight produced by upstream vertex-splitting). Ear-clipping such a loop directly produces a handful of
    triangles with a repeated corner, which are degenerate and get removed downstream, turning what should have
    been a shared internal edge back into boundary - i.e. hole-filling that nets *more* boundary edges than it
    closes. Splitting at every repeat into independent simple loops first avoids that class of triangle entirely."""
    seen: Dict[int, int] = {}
    for i, val in enumerate(loop.tolist()):
        if val in seen:
            j = seen[val]
            inner, outer = loop[j:i], np.concatenate([loop[:j], loop[i:]])
            out: List[np.ndarray] = []
            if len(inner) >= 3:
                out += _split_at_repeats(inner)
            if len(outer) >= 3:
                out += _split_at_repeats(outer)
            return out
        seen[val] = i
    return [loop]


def fill_holes(v: np.ndarray, f: np.ndarray, max_edges: int = 400) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    raw_loops = boundary_loops(f)
    new_faces: List[np.ndarray] = []
    filled = skipped = 0
    for raw in raw_loops:
        for loop in _split_at_repeats(raw):
            if len(loop) < 3 or len(loop) > max_edges:
                skipped += 1
                continue
            poly = loop[::-1]                                      # the fill traverses each boundary edge backwards
            pts = v[poly]
            tris = _ear_clip(pts)
            if not tris:
                skipped += 1
                continue
            new_faces.append(np.asarray([[poly[i], poly[j], poly[k]] for i, j, k in tris], np.int64))
            filled += 1
    if new_faces:
        f = np.concatenate([f] + new_faces, 0)
    return v, f, {"holes_filled": filled, "holes_skipped": skipped, "boundary_loops": len(raw_loops)}


# --------------------------------------------------------------------------------------------- islands
def remove_islands(v: np.ndarray, f: np.ndarray, min_faces: int, min_ratio: float):
    if len(f) == 0:
        return v, f, 0
    ncomp, lab = _components_faces(f)
    sizes = np.bincount(lab, minlength=ncomp)
    thr = max(min_faces, int(min_ratio * sizes.max()))
    keep_c = sizes >= thr
    if not keep_c.any():
        keep_c[np.argmax(sizes)] = True
    keep = keep_c[lab]
    removed = int((~keep_c).sum())
    v2, f2 = _compact(v, f[keep])
    return v2, f2, removed


# --------------------------------------------------------------------------------------------- driver
def is_watertight(f: np.ndarray) -> bool:
    if len(f) == 0:
        return False
    a, b, _, _ = _edge_table(f)
    nV = int(f.max()) + 1
    key = np.minimum(a, b) * nV + np.maximum(a, b)
    _, cnt = np.unique(key, return_counts=True)
    return bool((cnt == 2).all())


def topology_stats(f: np.ndarray) -> Dict[str, int]:
    if len(f) == 0:
        return {"faces": 0, "boundary_edges": 0, "nonmanifold_edges": 0}
    a, b, _, _ = _edge_table(f)
    nV = int(f.max()) + 1
    key = np.minimum(a, b) * nV + np.maximum(a, b)
    _, cnt = np.unique(key, return_counts=True)
    return {"faces": int(len(f)), "boundary_edges": int((cnt == 1).sum()), "nonmanifold_edges": int((cnt > 2).sum())}


def repair_mesh(verts: np.ndarray, faces: np.ndarray, opts: Optional[RepairOptions] = None):
    """Returns (verts, faces, report)."""
    o = opts or RepairOptions()
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    report: Dict[str, object] = {"before": topology_stats(f)}
    diag = float(np.linalg.norm(v.max(0) - v.min(0))) if len(v) else 1.0
    v, f = _compact(v, f)
    v, f, n_w = weld(v, f, o.weld_tol * diag)
    v, f, n_d = remove_degenerate(v, f, o.min_area * diag * diag)
    report.update(welded=n_w, degenerate_removed=n_d, holes_filled=0, hole_fill_passes=0)
    # Manifold-fixing, orientation and hole-filling can each reveal new work for the others (cutting a
    # non-manifold edge can open a boundary; filling a hole can touch itself and need re-cutting) - iterate the
    # group to a fixed point rather than assuming one pass of each suffices. A pathological loop (extremely
    # dense, near-degenerate input) can make hole-filling net *more* boundary than it closes each round; track
    # the best (fewest boundary + non-manifold edges) state seen and fall back to it if things get worse instead
    # of spinning for the full iteration budget or, worse, returning a result poorer than an earlier pass.
    best_v, best_f, best_score = v, f, float("inf")
    prev_faces = -1
    for it in range(8):
        if o.make_manifold and len(f):
            v, f, r = make_manifold(v, f)
            report["nonmanifold_edges"] = r["nonmanifold_edges"]
            report["vertices_split"] = report.get("vertices_split", 0) + r["vertices_split"]
            v, f, _n = remove_degenerate(v, f, o.min_area * diag * diag)
        if o.orient and len(f):
            f, n_flip = orient_consistently(v, f)
            report["faces_flipped"] = report.get("faces_flipped", 0) + n_flip
        if o.fill_holes and len(f) and topology_stats(f)["boundary_edges"]:
            v, f, r = fill_holes(v, f, o.max_hole_edges)
            report["holes_filled"] += r["holes_filled"]
            report["holes_skipped"] = r["holes_skipped"]
            report["boundary_loops"] = r["boundary_loops"]
            # NOT remove_degenerate here: an ear-clipped fill can be a thin-but-valid sliver, and stripping it by
            # area immediately after closing the loop reopens exactly the boundary this pass just closed. Let the
            # *next* iteration's manifold-fix/degenerate pass (or, on the last iteration, none) deal with it -
            # the best-state tracking below keeps this closed result if a later pass makes things worse instead.
            if o.orient:
                f, n_flip = orient_consistently(v, f)
                report["faces_flipped"] = report.get("faces_flipped", 0) + n_flip
        report["hole_fill_passes"] = it + 1
        st = topology_stats(f)
        score = st["boundary_edges"] + 4 * st["nonmanifold_edges"]
        if score < best_score:
            best_v, best_f, best_score = v, f, score
        if score == 0 or len(f) == prev_faces:
            break
        prev_faces = len(f)
    v, f = best_v, best_f
    if o.remove_islands and len(f):
        v, f, n_i = remove_islands(v, f, o.min_island_faces, o.min_island_ratio)
        report["islands_removed"] = n_i
    report["after"] = topology_stats(f)
    report["watertight"] = is_watertight(f)
    return v.astype(np.float32), f, report
