"""Turning what the person gave us (words, one picture, several pictures) into a `Request`.

Inputs are free-form: ``make "a bronze dragon"``, ``make knight.png``, ``make front.png back.png side.png`` or
``make front=a.png back=b.png left@80=c.png``.  A name like ``knight_back.png`` is understood, ``label=path`` is explicit,
and anything that is not an existing file is part of the text prompt.  Every picture goes through the multi-method outline
extractor (`outline.py`) and is framed identically; a picture whose outline cannot be trusted stops the run with a clear
message instead of being turned into a pasted slab.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .outline import Outline, OutlineError, extract_outline
from .views import ViewSpec, canonical_label, infer_label

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


@dataclass
class ViewInput:
    spec: ViewSpec
    image: Image.Image                 # square RGBA cut-out, framed
    mask: np.ndarray                   # bool, same size
    outline: Outline
    path: Optional[str] = None
    source: str = "user"               # user | generated


@dataclass
class Request:
    text: Optional[str] = None
    views: List[ViewInput] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if not self.views:
            return "text"
        return "multi" if len(self.views) > 1 else "image"

    def reference(self) -> Optional[ViewInput]:
        for v in self.views:
            if v.spec.label == "front":
                return v
        return self.views[0] if self.views else None

    def by_label(self, label: str) -> Optional[ViewInput]:
        return next((v for v in self.views if v.spec.label == label), None)


def split_inputs(items: Sequence[str]) -> Tuple[List[Tuple[Optional[str], Path]], List[str]]:
    """-> ([(label or None, path)], [text pieces])."""
    files: List[Tuple[Optional[str], Path]] = []
    words: List[str] = []
    for it in items:
        it = str(it)
        if "=" in it:
            left, _, right = it.partition("=")
            p = Path(right.strip().strip('"'))
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and left.strip() and " " not in left.strip():
                files.append((left.strip(), p))
                continue
        p = Path(it.strip().strip('"'))
        try:
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                files.append((None, p))
                continue
        except OSError:
            pass
        words.append(it)
    return files, words


def frame_square(rgba: Image.Image, ratio: float = 0.85) -> Image.Image:
    from ..image.preprocess import frame_foreground
    return frame_foreground(rgba, ratio)


def load_request(items: Sequence[str], text: Optional[str] = None, kind: str = "auto", fg_ratio: float = 0.85,
                 strict: bool = True, say: Optional[Callable[[str], None]] = None) -> Request:
    say = say or (lambda *_: None)
    files, words = split_inputs(items)
    prompt = " ".join(w for w in ([text] if text else []) + words if w and w.strip()).strip() or None
    req = Request(text=prompt)
    if not files and not prompt:
        raise ValueError("nothing to build from: give a text description, a picture, or several pictures")
    labelled: List[Tuple[ViewSpec, Path]] = []
    pending: List[Path] = []
    for lab, p in files:
        if lab is not None:
            labelled.append((ViewSpec.parse(lab), p))
            continue
        guess = infer_label(p.name)
        if guess:
            labelled.append((ViewSpec.parse(guess), p))
        else:
            pending.append(p)
    have = {s.label for s, _ in labelled}
    if pending:
        if len(pending) == 1 and "front" not in have:
            labelled.append((ViewSpec.parse("front"), pending[0]))
            req.notes.append(f"{pending[0].name}: no view name in the file name - taken as the front view")
        elif len(files) == 1:
            labelled.append((ViewSpec.parse("front"), pending[0]))
        else:
            raise ValueError("cannot tell which side these pictures show: " + ", ".join(p.name for p in pending)
                             + " - put front/back/left/right in the file name or write label=path (e.g. back=rear.png, "
                               "or an exact angle: view@135=photo.png)")
    seen = set()
    for spec, p in labelled:
        key = (round(spec.az, 3), round(spec.el, 3))
        if key in seen:
            raise ValueError(f"two pictures for the same view ({spec.label}): {p.name}")
        seen.add(key)
    labelled.sort(key=lambda t: (t[0].label != "front", abs(t[0].az)))
    for spec, p in labelled:
        img = Image.open(p)
        img.load()
        try:
            o = extract_outline(img, kind=kind, strict=strict, say=say)
        except OutlineError as e:
            raise OutlineError(f"{p.name}: {e}") from None
        framed = frame_square(o.rgba(), fg_ratio)
        mask = np.asarray(framed.getchannel("A")) >= 128
        for n in o.notes[1:]:
            req.notes.append(f"{p.name}: {n}")
        req.views.append(ViewInput(spec, framed, mask, o, str(p)))
    if req.views and req.reference() is None:
        raise ValueError("no usable picture")
    return req


def request_from_arrays(rgb: np.ndarray, alpha: np.ndarray, label: str = "front", text: Optional[str] = None,
                        fg_ratio: float = 0.85, method: str = "given") -> Request:
    """A one-picture request from a colour array and an alpha the caller already trusts (used by `asset` / `studio`)."""
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    hard = a >= 0.5
    o = Outline(a, hard, np.asarray(rgb[..., :3], np.uint8), "given", method, 1.0, {method: 1.0}, [])
    framed = frame_square(o.rgba(), fg_ratio)
    mask = np.asarray(framed.getchannel("A")) >= 128
    return Request(text=text, views=[ViewInput(ViewSpec.parse(label), framed, mask, o, None)])
