"""`aura-white setup`: fetch (or unpack) the upstream model repositories into third_party/.

The repositories are only *read* at run time (Aura White imports their model code); nothing is compiled."""
from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Sequence

REPOS = {
    "zero123plus": ("https://codeload.github.com/SUDO-AI-3D/zero123plus/zip/refs/heads/main", "zero123plus-main",
                    "diffusers-support/pipeline.py"),
    "instantmesh": ("https://codeload.github.com/TencentARC/InstantMesh/zip/refs/heads/main", "InstantMesh-main",
                    "src/models/lrm_mesh.py"),
    "hunyuan3d": ("https://codeload.github.com/Tencent-Hunyuan/Hunyuan3D-2.1/zip/refs/heads/main",
                  "Hunyuan3D-2.1-main", "hy3dshape/hy3dshape/pipelines.py"),
    "hunyuan3d2": ("https://codeload.github.com/Tencent-Hunyuan/Hunyuan3D-2/zip/refs/heads/main", "Hunyuan3D-2-main",
                   "hy3dgen/shapegen/pipelines.py"),
    "trellis2": ("https://codeload.github.com/microsoft/TRELLIS.2/zip/refs/heads/main", "TRELLIS.2-main",
                 "trellis2/pipelines/__init__.py"),
    "pixal3d": ("https://codeload.github.com/TencentARC/Pixal3D/zip/refs/heads/master", "Pixal3D-master",
                "inference.py"),
    "nvdiffrast": ("https://codeload.github.com/NVlabs/nvdiffrast/zip/refs/heads/main", "nvdiffrast-main",
                   "nvdiffrast/torch/ops.py"),
}


def _target(dir_: Optional[str]) -> Path:
    if dir_:
        return Path(dir_)
    return Path(__file__).resolve().parents[2] / "third_party"


def _extract(zpath: Path, dest: Path) -> Optional[Path]:
    with zipfile.ZipFile(zpath) as z:
        top = {n.split("/")[0] for n in z.namelist() if n and not n.startswith("__MACOSX")}
        skip = (".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webp")     # demo media only inflate the folder
        for n in z.namelist():
            if n.startswith("__MACOSX") or n.lower().endswith(skip):
                continue
            z.extract(n, dest)
    return dest / sorted(top)[0] if top else None


def setup_repos(dir_: Optional[str] = None, repos: Sequence[str] = ("hunyuan3d2",),
                zips: Sequence[str] = ()) -> int:
    dest = _target(dir_)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"third_party folder: {dest}")
    bad = 0
    for z in zips:
        zp = Path(z)
        if not zp.is_file():
            print(f"  skip {z}: not found")
            bad += 1
            continue
        try:
            print(f"  unpacking {zp.name} ...")
            _extract(zp, dest)
        except Exception as e:
            print(f"  failed to unpack {zp.name}: {e}")
            bad += 1
    if not zips:
        for name in repos:
            url, folder, marker = REPOS[name]
            if (dest / folder / marker).exists():
                print(f"  {name}: already present")
                continue
            print(f"  {name}: downloading {url}")
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    zp = Path(tmp) / "repo.zip"
                    with urllib.request.urlopen(url, timeout=120) as r, open(zp, "wb") as f:
                        shutil.copyfileobj(r, f)
                    _extract(zp, dest)
                print(f"  {name}: ok")
            except Exception as e:
                print(f"  {name}: FAILED ({type(e).__name__}: {e})")
                print("      download the zip in a browser and run:  aura-white setup --zip <file.zip>")
                bad += 1
    from .base import find_repo  # noqa: WPS433

    print("\nfound:")
    for name, (_, folder, marker) in REPOS.items():
        found = (dest / folder / marker).exists() or find_repo(folder.rsplit("-", 1)[0], marker) is not None
        print(f"  {'ok ' if found else 'no '} {name}")
    return 1 if bad else 0
