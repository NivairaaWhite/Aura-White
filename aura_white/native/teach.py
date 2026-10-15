"""Teachers: let other (open, permissively licensed) 3-D generators produce training data for Aura - safely.

Aura does not bundle or call any of them at inference.  A *teacher* is used once, offline, to turn a picture into a mesh;
the (picture, mesh) pair becomes a training record IF AND ONLY IF

  1. the teacher's licence allows using its output to train another model (licenses.require_teacher - Hunyuan3D is
     refused here, for good, because its licence forbids exactly this);
  2. the mesh passes the ground-truth-free gates (gates.verify): closed, real depth, a distinct back, sane colours, and
     it matches the picture it was made from.  A teacher's failures are not learned.

Because the heavy generators need a GPU that this package does not assume, a teacher is anything that maps a picture file
to a mesh file:

    CommandTeacher("trellis2", "python run_trellis2.py --image {image} --out {out}")      # your own GPU box / venv
    FunctionTeacher("triposr", lambda picture_path: (verts, faces, cloud))                 # a Python call

    dd = teach(teacher, pictures, "data/teacher_trellis2")             # -> a DataDir (same format as everything else)
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import geometry as G
from .data import DataDir, DataSpec, record_from_mesh
from .gates import GateConfig, TargetView, verify
from .licenses import Decision, LicenceError, require_teacher, resolve
from .render import fit_camera, prep_picture, render

PICTURE_EXT = (".png", ".jpg", ".jpeg", ".webp")
Mesh = Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]


class Teacher:
    key: str = ""
    decision: Optional[Decision] = None

    def generate(self, picture: Path, workdir: Path) -> Optional[Mesh]:
        raise NotImplementedError


class FunctionTeacher(Teacher):
    def __init__(self, key: str, fn: Callable[[Path], Optional[Mesh]], commercial: bool = True, accept_risk: bool = False):
        self.key = key.lower()
        self.decision = require_teacher(self.key, commercial, accept_risk)         # raises LicenceError at construction
        self.fn = fn

    def generate(self, picture: Path, workdir: Path) -> Optional[Mesh]:
        return self.fn(picture)


class CommandTeacher(Teacher):
    """Runs a command per picture.  `{image}` and `{out}` are replaced (quoted); the command must write a mesh to {out}
    (.glb / .obj / .ply).  Nothing is executed through a shell, so a file name cannot inject commands."""

    def __init__(self, key: str, command: str, timeout: float = 900.0, commercial: bool = True, accept_risk: bool = False):
        self.key = key.lower()
        self.decision = require_teacher(self.key, commercial, accept_risk)
        if "hunyuan" in command.lower():                       # a mislabelled teacher: the key says one thing, the command another
            raise LicenceError("the command mentions Hunyuan3D, whose licence forbids using its output to train another model; "
                               "label and licence must match the model that really runs (this check catches accidents, it is not tamper-proof)")
        self.command, self.timeout = command, float(timeout)

    def generate(self, picture: Path, workdir: Path) -> Optional[Mesh]:
        from ..asset.meshio import load_mesh_coloured

        out = Path(workdir) / (picture.stem + ".glb")
        argv = [a.replace("{image}", str(picture)).replace("{out}", str(out)) for a in shlex.split(self.command)]
        try:
            subprocess.run(argv, check=True, timeout=self.timeout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, OSError):
            return None
        if not out.exists():
            return None
        return load_mesh_coloured(out, n_samples=400_000)


# ------------------------------------------------------------------------------------------------ alignment
def align_to_picture(verts: np.ndarray, faces: np.ndarray, picture: np.ndarray, size: int = 96) -> Tuple[np.ndarray, float]:
    """Generators disagree about which way is up / front.  Try the 24 axis-aligned rotations and keep the one whose front
    silhouette best matches the picture's cut-out.  -> (rotated verts, IoU)."""
    from ..gen3d.views import light_mesh, proper_rotations

    lv, lf = light_mesh(np.asarray(verts, np.float64), np.asarray(faces, np.int64), 8000)
    tgt = picture[..., 3] > 0.5
    from PIL import Image

    tgt = np.asarray(Image.fromarray(tgt.astype(np.uint8) * 255).resize((size, size))) > 127
    best = (-1.0, np.eye(3))
    for R in proper_rotations():
        v = lv @ R.T
        a, _, _ = render(v, lf, None, 0.0, 0.0, size)
        m = a[..., 3] > 0.5
        u = (m | tgt).sum()
        iou = float((m & tgt).sum() / u) if u else 0.0
        if iou > best[0]:
            best = (iou, R)
    return np.asarray(verts, np.float64) @ best[1].T, best[0]


# ------------------------------------------------------------------------------------------------ the loop
def load_picture(path: Path, size: int = 64) -> Optional[np.ndarray]:
    """A picture file -> the model's input.  Needs a cut-out (alpha); a picture without one is skipped, never guessed at."""
    from PIL import Image

    im = Image.open(path)
    if im.mode not in ("RGBA", "LA", "P"):
        return None
    a = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0
    if a[..., 3].min() > 0.99:
        return None
    try:
        return prep_picture(a, size)
    except ValueError:
        return None


def teach(teacher: Teacher, pictures: Union[str, Path, Sequence[Union[str, Path]]], root, spec: Optional[DataSpec] = None,
          gate_cfg: Optional[GateConfig] = None, min_iou: float = 0.80, expect_solid: bool = True, limit: Optional[int] = None,
          seed: int = 0, min_sign_agreement: float = 0.8, say=print) -> DataDir:
    """pictures (a folder of cut-out PNGs, or a list) -> teacher meshes -> gates -> a DataDir.  Returns it, with a report in
    `dd.report` (how many were skipped and why).  `min_sign_agreement`: how well the mesh's inside/outside must agree between the
    normal test and the flood fill (thin, leaky or non-closed meshes disagree and are dropped)."""
    say = say or (lambda *_: None)
    if teacher.decision is None or not teacher.decision.allowed:
        raise LicenceError(f"teacher {teacher.key!r} has no licence decision")
    if isinstance(pictures, (str, Path)):
        files = sorted(p for p in Path(pictures).rglob("*") if p.suffix.lower() in PICTURE_EXT)
    else:
        files = [Path(p) for p in pictures]
    if limit:
        files = files[:limit]
    dd = DataDir(root, spec)
    rng = np.random.default_rng(seed)
    cfg = gate_cfg or GateConfig(size=96)
    why: Dict[str, int] = {}

    def skip(reason: str):
        why[reason] = why.get(reason, 0) + 1

    src = resolve(teacher.key)
    attribution = src.attribution if src else ""
    with tempfile.TemporaryDirectory(prefix="aura_teach_") as tmp:
        for k, p in enumerate(files):
            pic = load_picture(p, dd.spec.image_size)
            if pic is None:
                skip("picture has no cut-out (alpha)")
                continue
            try:
                mesh = teacher.generate(p, Path(tmp))
            except Exception as e:                                            # a crash in a teacher must not stop the run
                mesh = None
                say(f"  {p.name}: teacher failed ({type(e).__name__}: {e})")
            if mesh is None:
                skip("teacher produced no mesh")
                continue
            v, f, cloud = mesh
            v, iou = align_to_picture(v, f, pic)
            if iou < min_iou:
                skip(f"mesh does not match its own picture (IoU < {min_iou:g})")
                continue
            vn, fn, _, _ = G.normalize_mesh(v, f)
            colours = None                                                    # colour gate needs vertex colours; skipped without
            verdict = verify(vn, fn, colours, [TargetView(pic, 0.0, 0.0, "front")], expect_solid=expect_solid, cfg=cfg)
            hard = [x for x in verdict.failing if x in ("mesh", "depth", "fidelity")]
            if hard:
                skip("failed gates: " + ",".join(hard))
                continue
            rec = record_from_mesh(v, f, cloud, dd.spec, rng, source=teacher.key, licence=(src.licence if src else "?"),
                                   picture=pic, meta={"file": p.name, "gate_score": round(verdict.score, 3)})
            if rec is None or json.loads(str(rec["meta"]))["sign_agreement"] < min_sign_agreement:
                skip("mesh unusable (no area / leaky inside-outside)")
                continue
            dd.add(rec, p.stem, teacher.key, src.licence if src else "?", attribution, {"file": p.name, "gate_score": round(verdict.score, 3)})
            if k % 10 == 9:
                say(f"  teacher {teacher.key}: kept {len(dd)} of {k + 1}")
    dd.prov.add(teacher.key, 0, teacher.decision, attribution)                 # writes the licence decision into the manifest
    dd.save()
    dd.report = {"offered": len(files), "kept": len(dd), "skipped": why}       # type: ignore[attr-defined]
    return dd
