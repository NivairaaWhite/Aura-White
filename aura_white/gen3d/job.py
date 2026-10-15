"""A job is a folder.  Everything needed to refine a result later is inside it, and nothing is ever overwritten:
each build is a new version (v1, v2, ...) next to the others.

    <out>/<name>/
        job.json            request, recipe + signature, options, hardware, every version's metrics
        inputs/             the pictures used (cut-outs), including the chosen text-to-image reference
        views/              outline previews and the other text-to-image candidates
        v1/ v2/ ...         mesh.glb, report.json and the finished asset pack of that version
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def slug(text: str, fallback: str = "asset") -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return (s[:40] or fallback)


class Job:
    def __init__(self, root: Path, data: Optional[Dict[str, Any]] = None):
        self.root = Path(root)
        self.data: Dict[str, Any] = data or {"schema": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": []}

    @staticmethod
    def create(root, **fields) -> "Job":
        root = Path(root)
        for sub in ("inputs", "views"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        j = Job(root)
        j.data.update(fields)
        j.save()
        return j

    @staticmethod
    def load(root) -> "Job":
        root = Path(root)
        p = root / "job.json" if root.is_dir() else root
        if not p.is_file():
            raise FileNotFoundError(f"{root} is not an Aura White job folder (no job.json)")
        return Job(p.parent, json.loads(p.read_text(encoding="utf-8")))

    def save(self):
        (self.root / "job.json").write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")

    @property
    def versions(self) -> List[Dict[str, Any]]:
        return self.data.setdefault("versions", [])

    def next_version(self) -> str:
        n = 1 + max([int(v["id"][1:]) for v in self.versions] or [0])
        vid = f"v{n}"
        (self.root / vid).mkdir(parents=True, exist_ok=True)
        return vid

    def add_version(self, entry: Dict[str, Any]):
        self.versions.append(entry)
        self.data["latest"] = entry["id"]
        self.save()

    def latest(self) -> Optional[Dict[str, Any]]:
        return self.versions[-1] if self.versions else None
