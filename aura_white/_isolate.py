"""Run a function in a child Python process.

Compiled helper libraries (xatlas, mesh simplifiers, ...) are the usual reason a pipeline stops working
after a Python / NumPy update - and they fail by *crashing the interpreter*, which try/except cannot catch.
Calling them through this helper turns a segfault, heap corruption or hang into an ordinary exception so
the caller can fall back to the built-in pure-Python implementation."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from typing import Dict

import numpy as np


class IsolatedCallError(RuntimeError):
    pass


def call_isolated(target: str, arrays: Dict[str, object], timeout: float = 300.0) -> Dict[str, np.ndarray]:
    """target = "package.module:function"; the function receives the given arrays/scalars as keyword
    arguments and must return a dict of arrays."""
    with tempfile.TemporaryDirectory(prefix="aw_iso_") as tmp:
        src, dst = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(src, **{k: np.asarray(v) for k, v in arrays.items()})
        env = dict(os.environ)
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
        try:
            r = subprocess.run([sys.executable, "-m", "aura_white._isolate", target, src, dst],
                               capture_output=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            raise IsolatedCallError(f"{target} timed out after {timeout:.0f}s")
        if r.returncode != 0 or not os.path.exists(dst):
            tail = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            raise IsolatedCallError(f"{target} crashed (exit code {r.returncode}) {tail[0][:200]}")
        with np.load(dst) as d:
            return {k: d[k] for k in d.files}


def selftest_ok(x=1):
    return {"y": np.asarray(x) * 2}


def selftest_crash():                       # what a segfaulting native library looks like to the parent
    os.abort()


def _main(argv):
    target, src, dst = argv[1:4]
    mod, fn = target.split(":")
    with np.load(src) as d:
        kwargs = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
    out = getattr(importlib.import_module(mod), fn)(**kwargs)
    np.savez(dst, **out)


if __name__ == "__main__":
    _main(sys.argv)
