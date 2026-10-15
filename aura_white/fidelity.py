"""Reference-locked 3D reconstruction.

This mode is intentionally different from a semantic image-to-3D model: it treats the supplied image as the
non-negotiable reference for the photographed view.  The front surface is a triangulation of the *exact* foreground
mask (every island, every hole), its UVs are the image's own pixel coordinates, and the generated depth only gives
that surface real 3D thickness.  Seen from the reference camera the result therefore reproduces the source pixels.

How exactness is obtained (all NumPy/SciPy, no optional dependency decides the outcome):

1. Boundary vertices sit on the 0.5 iso-line of the mask: one vertex at the centre of every foreground/background
   pixel edge.  That is order-free, so it needs no contour tracing and cannot drop islands, holes or hair-thin
   chains the way "largest polygon only" contour extraction does.
2. A Delaunay triangulation of boundary + interior samples is filtered by where each triangle really lies.
3. Render-compare-correct: the triangulation is rasterised at working resolution, every mask pixel it does not
   cover receives a Steiner vertex, and the triangulation is rebuilt - until the coverage target is met.
4. The closed shell is built from *directed* boundary edges, so side walls wind consistently by construction and no
   heavyweight mesh repair is needed (it is only a fallback if the manifold check fails).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Delaunay

from .forge.softraster import rasterize, render_camera, unproject
from .mesh.cleanup import signed_volume
from .mesh.repair import RepairOptions, repair_mesh


@dataclass
class LockedMesh:
    verts: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    report: Dict[str, object]


# ------------------------------------------------------------------------------------------------ helpers
def _boundary_points(mask: np.ndarray) -> np.ndarray:
    """Centre of every foreground/background pixel edge = the 0.5 iso-line of the mask.
    Pixel (y, x) spans [x, x+1] x [y, y+1], so its centre is (x + 0.5, y + 0.5)."""
    H, W = mask.shape
    p = np.pad(mask, 1, constant_values=False)                        # p[y + 1, x + 1] == mask[y, x]
    ys, js = np.nonzero(p[1:H + 1, :-1] != p[1:H + 1, 1:])            # vertical edge between column j-1 and j
    ivs, xs = np.nonzero(p[:-1, 1:W + 1] != p[1:, 1:W + 1])           # horizontal edge between row i-1 and i
    return np.concatenate([np.stack([js.astype(np.float64), ys + 0.5], 1),
                           np.stack([xs + 0.5, ivs.astype(np.float64)], 1)], 0)


def _jitter(points: np.ndarray, amplitude: float = 2e-5) -> np.ndarray:
    """Deterministic sub-micro-pixel jitter: pixel-edge vertices are exactly collinear and co-circular, which makes
    Delaunay ambiguous (sliver triangles whose orientation flips).  Jittered points are then used for *everything*
    (triangulation, UVs, unprojection), so topology and geometry can never disagree."""
    rng = np.random.RandomState(12345)
    return points + rng.uniform(-amplitude, amplitude, points.shape)


def _accept(points: np.ndarray, tri: np.ndarray, mask: np.ndarray, mask_grow: np.ndarray) -> np.ndarray:
    """Keep triangles whose centroid is inside the mask and whose edge midpoints are inside the (1px grown) mask."""
    H, W = mask.shape
    p = points[tri]
    c = p.mean(1)
    cy = np.clip(np.floor(c[:, 1]).astype(np.int64), 0, H - 1)
    cx = np.clip(np.floor(c[:, 0]).astype(np.int64), 0, W - 1)
    ok = mask[cy, cx]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        q = (p[:, a] + p[:, b]) * 0.5
        qy = np.clip(np.floor(q[:, 1]).astype(np.int64), 0, H - 1)
        qx = np.clip(np.floor(q[:, 0]).astype(np.int64), 0, W - 1)
        ok &= mask_grow[qy, qx]
    return ok


def _orient_ccw(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p = points[tri]
    area = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    tri = tri.copy()
    neg = area < 0
    tri[neg] = tri[neg][:, ::-1]
    return tri[np.abs(area) > 1e-9]


def _triangulate(points: np.ndarray, mask: np.ndarray, mask_grow: np.ndarray) -> np.ndarray:
    tri = Delaunay(points).simplices.astype(np.int64)       # points are in general position (see _jitter)
    tri = tri[_accept(points, tri, mask, mask_grow)]
    return _orient_ccw(points, tri)


def _coverage(points: np.ndarray, tri: np.ndarray, mask: np.ndarray):
    """Rasterise the front triangulation in image space; drop triangles that mostly cover background.
    Returns (kept triangles, covered (H,W) bool)."""
    H, W = mask.shape
    tid, _, _ = rasterize(points, np.ones(len(points)), tri, H, W)
    hit = tid >= 0
    total = np.bincount(tid[hit], minlength=len(tri)).astype(np.float64)
    inside = np.bincount(tid[hit & mask], minlength=len(tri)).astype(np.float64)
    keep = (total == 0) | (inside >= 0.5 * np.maximum(total, 1.0))
    covered = hit & keep[np.where(hit, tid, 0)]
    return tri[keep], covered


def _split_pinch_vertices(points: np.ndarray, tri: np.ndarray):
    """Give every fan of triangles around a vertex its own copy of that vertex.

    Where the mask touches itself (thin necks, chain links) two triangle fans share one vertex; extruding that into a
    shell makes an edge with four faces.  Corners of triangles joined through a shared edge belong to one fan, so the
    connected components of that corner graph are exactly the vertex copies a manifold surface needs."""
    T, n = len(tri), len(points)
    t_id = np.repeat(np.arange(T), 3)
    k_id = np.tile(np.arange(3), T)
    a = tri[t_id, k_id]
    b = tri[t_id, (k_id + 1) % 3]
    fwd = a * n + b
    order = np.argsort(fwd)
    srt = fwd[order]
    want = b * n + a
    pos = np.clip(np.searchsorted(srt, want), 0, len(srt) - 1)
    hit = srt[pos] == want
    i = np.flatnonzero(hit)                                  # directed edge i = (t1, k1): a -> b
    j = order[pos[hit]]                                      # its opposite   j = (t2, k2): b -> a
    node = lambda t, k: 3 * t + k                            # noqa: E731  (corner k of triangle t)
    r = np.concatenate([node(t_id[i], k_id[i]), node(t_id[i], (k_id[i] + 1) % 3)])
    c = np.concatenate([node(t_id[j], (k_id[j] + 1) % 3), node(t_id[j], k_id[j])])
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(3 * T, 3 * T))
    ncomp, label = connected_components(g, directed=False)
    corner_vertex = tri.reshape(-1)                          # original vertex of corner node 3*t + k
    new_points = np.empty((ncomp, 2), np.float64)
    new_points[label] = points[corner_vertex]
    orig = np.empty(ncomp, np.int64)
    orig[label] = corner_vertex
    # Copies of one pinch vertex would sit at the same position, so welding by position (Blender, trimesh, slicers)
    # would glue the two walls together again into an edge shared by four faces.  Nudge each copy a hundredth of a
    # pixel towards the centre of its own fan: invisible, but the exported mesh stays manifold after any weld.
    copies = np.bincount(orig, minlength=len(points))[orig]
    pinch = copies > 1
    if pinch.any():
        tc = points[tri].mean(1)                                       # triangle centres (T, 2)
        lab = label.reshape(T, 3)
        cx = np.bincount(lab.ravel(), np.repeat(tc[:, 0], 3), minlength=ncomp)
        cy = np.bincount(lab.ravel(), np.repeat(tc[:, 1], 3), minlength=ncomp)
        cnt = np.maximum(np.bincount(lab.ravel(), minlength=ncomp), 1)
        d = np.stack([cx / cnt, cy / cnt], 1) - new_points
        norm = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        new_points = np.where(pinch[:, None], new_points + 0.01 * d / norm, new_points)
    return new_points, label.reshape(T, 3).astype(np.int64), int(ncomp - len(np.unique(tri)))


def _closed_shell(tri: np.ndarray, n: int):
    """Front + back + side walls.  `tri` must be consistently oriented; side walls follow *directed* boundary edges.
    Vertex i is the front copy and i + n the back copy of the same image point."""
    de = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], 0)
    key = de[:, 0] * n + de[:, 1]
    rev = de[:, 1] * n + de[:, 0]
    boundary = de[~np.isin(key, rev)]
    u, v = boundary[:, 0], boundary[:, 1]
    side = np.empty((len(boundary) * 2, 3), np.int64)
    side[0::2] = np.stack([v, u, u + n], 1)
    side[1::2] = np.stack([v, u + n, v + n], 1)
    return np.concatenate([tri, tri[:, ::-1] + n, side], 0), 2 * len(tri)


def _wall_uv_split(verts: np.ndarray, faces: np.ndarray, uv: np.ndarray, points: np.ndarray, mask: np.ndarray,
                   n: int, cap_faces: int, W: int, H: int):
    """Side walls get their own vertex copies whose UVs point a few pixels *inside* the artwork.

    The caps must sample the picture exactly at the silhouette (that is what makes the reference view identical),
    but the same rim texel painted along a wall shows the anti-aliased edge - a pale streak on every side view.
    Positions stay identical, so the shell is still closed; only the texture seam differs.
    `faces[cap_faces:]` are the walls; vertex i is the front and i + n the back copy of image point i."""
    from scipy.ndimage import maximum_filter
    wall = faces[cap_faces:]
    if len(wall) == 0:
        return verts, faces, uv
    dist = distance_transform_edt(mask)
    deep = dist >= np.clip(0.6 * maximum_filter(dist, size=9), 1.5, 4.0)   # thin parts: 60% of their half-width
    if not deep.any():
        return verts, faces, uv
    _, (iy, ix) = distance_transform_edt(~deep, return_distances=True, return_indices=True)
    used = np.unique(wall)
    base = np.where(used < n, used, used - n)
    px = np.clip(np.floor(points[base, 0]).astype(np.int64), 0, mask.shape[1] - 1)
    py = np.clip(np.floor(points[base, 1]).astype(np.int64), 0, mask.shape[0] - 1)
    inward = np.stack([(ix[py, px] + 0.5) / W, (iy[py, px] + 0.5) / H], 1).astype(np.float32)
    remap = np.full(2 * n, -1, np.int64)
    remap[used] = len(verts) + np.arange(len(used))
    out = faces.copy()
    out[cap_faces:] = remap[wall]
    return np.concatenate([verts, verts[used]], 0), out, np.concatenate([uv, inward], 0)


def _manifold_stats(faces: np.ndarray) -> Tuple[int, int]:
    """(open edges, over-shared edges) counted on undirected edges."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def _defringe(rgb: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Remove the old-background blend from the outermost ~1 px of the artwork.

    Only pixels that are pale and nearly grey *and* clearly brighter than the colour a couple of pixels further in are
    replaced by that inner colour, so genuinely white/silver parts (whose neighbours are white too) are untouched."""
    from scipy.ndimage import maximum_filter
    dist = distance_transform_edt(ok)
    deep = dist >= np.clip(0.6 * maximum_filter(dist, size=9), 2.0, 4.0)
    if not deep.any():
        return rgb
    d_deep, (iy, ix) = distance_transform_edt(~deep, return_distances=True, return_indices=True)
    inner = rgb[iy, ix]
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    lum_in = inner @ np.array([0.299, 0.587, 0.114], np.float32)
    pale = (rgb.max(-1) - rgb.min(-1)) < 0.2
    fringe = ok & (dist <= 1.5) & (d_deep <= 4.0) & pale & (lum - lum_in > 0.12)
    return np.where(fringe[..., None], inner, rgb)


def source_texture(rgb: np.ndarray, alpha: np.ndarray, max_side: int = 4096, solid: float = 0.5,
                   defringe: bool = False) -> np.ndarray:
    """Texture for the locked surface: the source pixels, unchanged wherever the surface exists (alpha >= `solid`).

    `solid` = 0.5 matches the geometry boundary (the mesh outline is the 0.5 iso-line of the mask), so every pixel the
    mesh can show is the original pixel and the reference view is identical.  Everything outside is replaced by the
    nearest kept colour so bilinear sampling at the silhouette cannot blend in the old background.  A higher `solid`
    (0.98) plus `defringe` additionally replaces the soft anti-aliased band - cleaner on dark viewers, not pixel-exact.

    The texture keeps the picture's own width x height (UVs are its pixel coordinates), so the reference view samples it
    1:1 with no resampling loss; it is only scaled down if its longest side exceeds `max_side`.  uint8 (h, w, 3)."""
    from PIL import Image
    rgb = np.asarray(rgb, np.float32)
    ok = np.asarray(alpha, np.float32) >= solid
    if defringe and ok.any() and not ok.all():
        rgb = _defringe(rgb, ok)
    if ok.any() and not ok.all():
        _, (iy, ix) = distance_transform_edt(~ok, return_distances=True, return_indices=True)
        rgb = np.where(ok[..., None], rgb, rgb[iy, ix])
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    h, w = rgb.shape[:2]
    k = min(1.0, float(max_side) / max(h, w))
    if k < 1.0:
        img = img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.LANCZOS)
    return np.asarray(img)


# ------------------------------------------------------------------------------------------------- builder
def build_reference_locked_mesh(ref_alpha: np.ndarray, source_verts: np.ndarray, source_faces: np.ndarray,
                                camera, working_res: int = 2048, thickness_ratio: float = 0.02,
                                point_step: int = 8, boundary_step: int = 1, max_repair_passes: int = 5,
                                coverage_target: float = 0.99995, wall_uv: bool = True) -> LockedMesh:
    """Build a closed 3-D shell whose photographed silhouette equals the input mask.

    `working_res` is the longest side of the working mask; it is never upscaled, and images up to it are used at
    their native size (down-sampling is what used to erase thin chains).  `boundary_step` > 1 thins the boundary
    vertices (smaller mesh, still repaired by the coverage loop); 1 keeps every iso-line vertex.
    """
    a = np.asarray(ref_alpha, np.float32)
    H0, W0 = a.shape
    S = int(max(128, working_res))
    scale = min(1.0, S / max(H0, W0))
    H, W = max(32, int(round(H0 * scale))), max(32, int(round(W0 * scale)))
    if (H, W) != (H0, W0):
        from PIL import Image
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR),
                       np.float32) / 255.0
    mask = a >= 0.5
    if mask.sum() < 32:
        raise ValueError("reference image has too little foreground for source-locked reconstruction")

    from scipy.ndimage import binary_dilation
    mask_grow = binary_dilation(mask, structure=np.ones((3, 3), bool))

    pts_b = _boundary_points(mask)
    if len(pts_b) == 0:
        raise ValueError("could not extract a foreground contour")
    if boundary_step > 1:
        pts_b = pts_b[:: int(boundary_step)]
    dist = distance_transform_edt(mask)
    gy, gx = np.mgrid[0:H:max(1, int(point_step)), 0:W:max(1, int(point_step))]
    keep = dist[gy, gx] >= max(1.5, 0.35 * point_step)
    pts_i = np.stack([gx[keep] + 0.5, gy[keep] + 0.5], 1)
    base = np.unique(np.round(np.concatenate([pts_b, pts_i], 0), 4), axis=0)
    if len(base) < 3:
        raise ValueError("could not sample the reference silhouette")
    points = _jitter(base)

    passes, history = 0, []
    while True:
        tri = _triangulate(points, mask, mask_grow)
        if len(tri) == 0:
            raise ValueError("reference silhouette triangulation produced no valid faces")
        tri, covered = _coverage(points, tri, mask)
        missing = mask & ~covered
        extra = covered & ~mask
        iou = float((covered & mask).sum() / max((covered | mask).sum(), 1))
        history.append(dict(pass_=passes, iou=round(iou, 5), missing_px=int(missing.sum()), extra_px=int(extra.sum())))
        if iou >= coverage_target or passes >= max_repair_passes or not missing.any():
            break
        my, mx = np.nonzero(missing)
        if len(my) > 80_000:                                # keep the mesh bounded on pathological inputs
            sel = np.linspace(0, len(my) - 1, 80_000).astype(np.int64)
            my, mx = my[sel], mx[sel]
        base = np.unique(np.round(np.concatenate([base, np.stack([mx + 0.5, my + 0.5], 1)], 0), 4), axis=0)
        points = _jitter(base)
        passes += 1

    # one vertex copy per triangle fan (this also drops vertices no kept triangle uses), then measure depth
    points, tri, n_split = _split_pinch_vertices(points, tri)

    sm, depth, _ = render_camera(camera, np.asarray(source_verts, np.float64), np.asarray(source_faces, np.int64), H, W)
    valid = np.isfinite(depth) & sm
    if not valid.any():
        raise ValueError("source mesh has no usable depth in the reference camera")
    _, nearest = distance_transform_edt(~valid, return_distances=True, return_indices=True)
    iy, ix = nearest
    depth_filled = depth.copy()
    depth_filled[~valid] = depth[iy[~valid], ix[~valid]]
    px = np.clip(np.floor(points[:, 0]).astype(np.int64), 0, W - 1)
    py = np.clip(np.floor(points[:, 1]).astype(np.int64), 0, H - 1)
    d = depth_filled[py, px].astype(np.float64)
    finite = d[np.isfinite(d)]
    d[~np.isfinite(d)] = float(np.median(finite))
    front = unproject(camera, points[:, 0], points[:, 1], d, H, W)
    # slab thickness is a fraction of the object's largest extent (the hidden sides are invented, so keep them slim);
    # the back is the front shifted along each view ray, so the two never intersect
    thickness = max(float(np.ptp(front, axis=0).max()) * float(thickness_ratio), 1e-3)
    back = unproject(camera, points[:, 0], points[:, 1], d + thickness, H, W)

    n = len(points)
    verts = np.concatenate([front, back], 0).astype(np.float32)
    uv0 = np.stack([points[:, 0] / W, points[:, 1] / H], 1).astype(np.float32)
    uv = np.concatenate([uv0, uv0], 0)
    faces, cap_faces = _closed_shell(tri, n)
    if signed_volume(verts, faces) < 0:                     # outward-facing normals
        faces = faces[:, ::-1].copy()

    open_edges, shared_edges = _manifold_stats(faces)
    repair_info: Dict[str, object] = {}
    if open_edges or shared_edges:                          # fallback only; the directed construction is normally closed
        verts, faces, rep = repair_mesh(
            verts, faces, RepairOptions(fill_holes=False, remove_islands=False, weld_tol=1e-9))
        from scipy.spatial import cKDTree
        _, nn = cKDTree(np.concatenate([front, back], 0)).query(verts.astype(np.float64), k=1, workers=-1)
        uv = uv[nn]
        repair_info = {k: v for k, v in rep.items() if k not in ("before", "after")}
        open_edges, shared_edges = _manifold_stats(faces)
    n_seam = 0
    if wall_uv and not repair_info:
        n_before = len(verts)
        verts, faces, uv = _wall_uv_split(verts, faces, uv, points, mask, n, cap_faces, W, H)
        n_seam = len(verts) - n_before

    report = {
        "mode": "source_locked", "working_resolution": [int(W), int(H)],
        "sampled_vertices_front": int(n), "faces": int(len(faces)),
        "thickness_view_depth": float(thickness),
        "source_locked_contract": "reference-view silhouette + source pixels are authoritative",
        "closed": bool(open_edges == 0 and shared_edges == 0), "open_edges": open_edges, "shared_edges": shared_edges,
        "pinch_vertices_split": n_split, "wall_uv_inward": bool(n_seam), "wall_uv_seam_vertices": int(n_seam),
        "closed_note": "closed = every edge has two faces before wall texture seams; seam vertices share positions",
        "mask_coverage_iou": history[-1]["iou"], "coverage_passes": history, "repair": repair_info,
    }
    return LockedMesh(verts, faces, uv, report)
