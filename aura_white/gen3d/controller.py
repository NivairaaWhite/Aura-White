"""The quality loop: generate -> orient -> fit to the pictures -> fuse -> audit -> score -> decide what to try next.

Nothing here knows about any particular neural network; a generator is just ``call(GenCall) -> (verts, faces)``.  That
keeps the whole decision logic testable and means a new backend needs no change here.

What it does that a single generator call cannot:
  * orients the mesh (generators disagree about which way is up / front) by testing the 24 axis rotations against the
    pictures - never by trusting a convention;
  * discovers swapped left/right conventions in multi-picture input and re-runs with the labels exchanged;
  * judges a candidate by how *three-dimensional and consistent* it is, not only by how well its front matches - a slab
    matches the front perfectly, which is why silhouette-IoU alone cannot be the score;
  * refines the winner against every picture (`fusion.fuse`), then decides from the measured failure what to change
    (new seeds / stronger guidance for a flat result, finer mesh for a missed outline).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import meshutil as M
from .audit import Audit, audit_mesh
from .fusion import ViewConstraint, build_constraints, fuse, view_ious
from .recipe import Recipe
from .views import ViewSpec, fit_cam, light_mesh, proper_rotations


@dataclass(frozen=True)
class GenCall:
    seed: int
    octree: int
    steps: int
    guidance: float
    label_swap: bool = False
    round: int = 0
    index: int = 0


@dataclass
class Candidate:
    verts: np.ndarray
    faces: np.ndarray
    call: GenCall
    audit: Audit
    ious: Dict[str, float]
    score: float
    fused: bool = False
    rotation: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def mean_iou(self) -> float:
        return float(np.mean(list(self.ious.values()))) if self.ious else 1.0

    def summary(self) -> Dict:
        return {"round": self.call.round, "index": self.call.index, "seed": self.call.seed, "score": round(self.score, 4),
                "view_iou": round(self.mean_iou, 4), "per_view": {k: round(v, 4) for k, v in self.ious.items()},
                "flat_risk": round(self.audit.flat_risk, 3), "depth_ratio": round(self.audit.depth_ratio, 3),
                "profile": self.audit.profile, "topology": round(self.audit.topology_score, 3), "fused": self.fused,
                "octree": self.call.octree, "guidance": self.call.guidance, "label_swap": self.call.label_swap, "notes": self.notes}


@dataclass
class LoopResult:
    best: Candidate
    passed: bool
    reason: str
    history: List[Dict]


ViewMask = Tuple[ViewSpec, np.ndarray]


def swapped(views: Sequence[ViewMask]) -> List[ViewMask]:
    """The same pictures with left and right exchanged."""
    out = []
    for spec, m in views:
        if spec.label == "left":
            out.append((ViewSpec("right", -spec.az, spec.el), m))
        elif spec.label == "right":
            out.append((ViewSpec("left", -spec.az, spec.el), m))
        else:
            out.append((ViewSpec(spec.label, -spec.az if abs(spec.az) not in (0.0, 180.0) else spec.az, spec.el), m))
    return out


def align_frame(v: np.ndarray, f: np.ndarray, views: Sequence[ViewMask], margin: float = 0.02,
                size: int = 160) -> Tuple[np.ndarray, int, float]:
    """Rotate the mesh by the one of the 24 axis rotations that best matches the pictures.  The identity wins ties (a
    generator's own convention is the best prior), so a symmetric silhouette does not flip the object."""
    rots = proper_rotations()
    ident = next(i for i, R in enumerate(rots) if np.allclose(R, np.eye(3)))
    lv, lf = light_mesh(v, f, 8_000)
    front = views[0]
    quick = []
    for i, R in enumerate(rots):
        _, iou, _ = fit_cam(lv @ R.T, lf, front[0].az, front[0].el, front[1], size, refine=False)
        quick.append((iou, i))
    cand = {i for _, i in sorted(quick, reverse=True)[:6]} | {ident}
    scores = {}
    for i in cand:
        ious = [fit_cam(lv @ rots[i].T, lf, s.az, s.el, m, size, refine=False)[1] for s, m in views]
        scores[i] = float(np.mean(ious))
    best = max(scores, key=lambda i: (scores[i], i == ident))
    if scores[ident] >= scores[best] - margin:
        best = ident
    return v @ rots[best].T, best, scores[best]


def side_swap_gain(v: np.ndarray, f: np.ndarray, views: Sequence[ViewMask], size: int = 160) -> float:
    """> 0 when the left/right pictures fit better with the labels exchanged."""
    sides = [(s, m) for s, m in views if s.label in ("left", "right")]
    if not sides:
        return 0.0
    lv, lf = light_mesh(v, f, 8_000)
    a = np.mean([fit_cam(lv, lf, s.az, s.el, m, size, refine=False)[1] for s, m in sides])
    b = np.mean([fit_cam(lv, lf, s.az, s.el, m, size, refine=False)[1] for s, m in swapped(sides)])
    return float(b - a)


def score_candidate(audit: Audit, ious: Dict[str, float], recipe: Recipe) -> float:
    view = float(np.mean(list(ious.values()))) if ious else 1.0
    solid = 1.0 - audit.flat_risk
    w = recipe.w_view + recipe.w_solid + recipe.w_topology
    return (recipe.w_view * view + recipe.w_solid * solid + recipe.w_topology * audit.topology_score) / w


def passes(c: Candidate, recipe: Recipe) -> bool:
    return (c.mean_iou >= recipe.view_iou_target and c.audit.flat_risk <= 0.35 and c.audit.topology_score >= 0.6)


def evaluate(v: np.ndarray, f: np.ndarray, call: GenCall, views: Sequence[ViewMask], recipe: Recipe,
             say: Callable[[str], None]) -> Candidate:
    notes: List[str] = []
    v, f = M.orient_outward(v, f)
    v, rot, fit = align_frame(v, f, views)
    if rot:
        notes.append(f"re-oriented the generator's axes (rotation #{rot}, front fit {fit:.2f})")
    cons = build_constraints(v, f, views)
    ious = view_ious(v, f, cons)
    fused = False
    if len(views) >= 2 or float(np.mean(list(ious.values()))) < recipe.view_iou_target:
        r = fuse(v, f, cons, grid=recipe.fusion_grid, tol_px=recipe.fusion_tolerance_px, say=say)
        notes += r.notes
        if r.applied:
            v, f, fused = r.verts, r.faces, True
            ious = view_ious(v, f, cons)
    audit = audit_mesh(v, f, recipe.min_depth_ratio, recipe.expect_solid)
    return Candidate(v, f, call, audit, ious, score_candidate(audit, ious, recipe), fused, rot, notes + audit.notes)


def run_quality_loop(generate: Callable[[GenCall], Tuple[np.ndarray, np.ndarray]], views: Sequence[ViewMask], recipe: Recipe,
                     base_seed: int = 0, say: Optional[Callable[[str], None]] = None,
                     allow_swap_probe: bool = True) -> LoopResult:
    say = say or (lambda *_: None)
    views = list(views)
    swap = False
    octree, guidance = recipe.octree, recipe.shape_guidance
    history: List[Dict] = []
    best: Optional[Candidate] = None
    reason = ""
    for rnd in range(max(1, recipe.max_rounds)):
        round_best: Optional[Candidate] = None
        for i in range(max(1, recipe.candidates)):
            call = GenCall(base_seed + 1000 * rnd + i, octree, recipe.shape_steps, guidance, swap, rnd, i)
            say(f"round {rnd + 1}/{recipe.max_rounds}, candidate {i + 1}/{recipe.candidates} (seed {call.seed})")
            v, f = generate(call)
            cand = evaluate(v, f, call, views, recipe, say)
            if allow_swap_probe and not swap and any(s.label in ("left", "right") for s, _ in views):
                gain = side_swap_gain(cand.verts, cand.faces, views)
                if gain > 0.05:
                    cand.notes.append(f"left/right pictures fit better swapped (+{gain:.2f} IoU): the generator uses the opposite convention")
                    swap = True
            history.append(cand.summary())
            if round_best is None or cand.score > round_best.score:
                round_best = cand
            if best is None or cand.score > best.score:
                best = cand
            say(f"  score {cand.score:.3f} (view IoU {cand.mean_iou:.3f}, depth {cand.audit.depth_ratio:.2f}, "
                f"flat-risk {cand.audit.flat_risk:.2f}{', fused' if cand.fused else ''})")
            if passes(cand, recipe) and not swap:
                reason = f"passed in round {rnd + 1}"
                return LoopResult(cand, True, reason, history)
        assert round_best is not None
        # decide what to change from what was measured
        if swap and any("opposite convention" in n for n in round_best.notes):
            say("  -> exchanging the left/right labels for the next round")
            continue
        if round_best.audit.flat_risk > 0.35:
            guidance = min(guidance + 1.5, 12.0)
            say(f"  -> flat result: stronger guidance ({guidance:g}) and new seeds")
        elif round_best.mean_iou < recipe.view_iou_target:
            octree = min(int(round(octree * 1.25 / 8.0)) * 8, 640)
            say(f"  -> outline off by {1 - round_best.mean_iou:.1%}: finer mesh grid ({octree}) and new seeds")
    assert best is not None
    why = []
    if best.mean_iou < recipe.view_iou_target:
        why.append(f"outline agreement {best.mean_iou:.3f} is below the target {recipe.view_iou_target}")
    if best.audit.flat_risk > 0.35:
        why.append(f"the mesh is flat ({best.audit.profile}, depth {best.audit.depth_ratio:.0%} of width)")
    if best.audit.topology_score < 0.6:
        why.append("the mesh has open edges or loose fragments")
    ok = not why
    return LoopResult(best, ok, "passed" if ok else "best of %d candidate(s); %s" % (len(history), "; ".join(why)), history)
