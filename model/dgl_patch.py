"""
Fix DGL 2.1 on Kaggle Py3.12 / PyTorch 2.6 for Struct2GO training.

Struct2GO uses dgl.dataloading.GraphDataLoader only — not graphbolt.dataloader.
Patches:
  1. graphbolt/base.py  — IterDataPipe from torch.utils.data
  2. graphbolt/__init__.py — skip dataloader import (needs torchdata.dataloader2)
"""
from __future__ import annotations

import importlib
import shutil
import site
import subprocess
import sys
from pathlib import Path

_PATCHES: list[tuple[str, str, str]] = [
    (
        "graphbolt/base.py",
        "from torchdata.datapipes.iter import IterDataPipe",
        "from torch.utils.data import IterDataPipe",
    ),
    (
        "graphbolt/__init__.py",
        "from .dataloader import *",
        "# from .dataloader import *  # dgl_patch: skip (torchdata.dataloader2); Struct2GO OK",
    ),
]


def _dgl_roots() -> list[Path]:
    return [
        Path(sp) / "dgl"
        for sp in site.getsitepackages()
        if (Path(sp) / "dgl" / "__init__.py").is_file()
    ]


def _clear_pycache(py: Path) -> None:
    cache = py.parent / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)


def _purge_modules(prefix: str) -> None:
    for key in list(sys.modules):
        if key == prefix or key.startswith(prefix + "."):
            del sys.modules[key]


def patch_dgl_on_disk(verbose: bool = True) -> int:
    n = 0
    for root in _dgl_roots():
        for rel, old, new in _PATCHES:
            py = root / rel
            if not py.is_file():
                continue
            text = py.read_text(encoding="utf-8")
            if old in text:
                py.write_text(text.replace(old, new), encoding="utf-8")
                _clear_pycache(py)
                n += 1
                if verbose:
                    print(f"[dgl_patch] {py}")
            elif new in text:
                if verbose:
                    print(f"[dgl_patch] already OK {py}")
            elif verbose:
                print(f"[dgl_patch] WARN no match in {py}")
    return n


def ensure_torchdata(verbose: bool = True) -> None:
    """Real torchdata package required; remove any fake shim from sys.modules."""
    _purge_modules("torchdata")
    importlib.invalidate_caches()

    try:
        td = importlib.import_module("torchdata")
        if not getattr(td, "__path__", None):
            raise ImportError(f"torchdata is not a package: {td!r}")
        importlib.import_module("torchdata.dataloader2.graph")
        if verbose:
            print(f"[dgl_patch] torchdata {getattr(td, '__version__', '?')} @ {td.__path__}")
        return
    except ImportError as exc:
        if verbose:
            print(f"[dgl_patch] installing torchdata==0.11.0 ({exc})")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "torchdata==0.11.0"],
        check=True,
    )
    _purge_modules("torchdata")
    importlib.invalidate_caches()
    td = importlib.import_module("torchdata")
    if not getattr(td, "__path__", None):
        raise ImportError("torchdata install failed — still not a package")
    importlib.import_module("torchdata.dataloader2.graph")
    if verbose:
        print(f"[dgl_patch] torchdata {td.__version__} installed")


def ensure_dgl_importable(verbose: bool = True) -> None:
    ensure_torchdata(verbose=verbose)
    patch_dgl_on_disk(verbose=verbose)
    _purge_modules("dgl")
    importlib.invalidate_caches()
    dgl = importlib.import_module("dgl")

    if verbose:
        import torch

        print(
            f"[dgl_patch] OK  DGL {dgl.__version__}  torch {torch.__version__}"
        )


def main() -> None:
    ensure_dgl_importable(verbose=True)


if __name__ == "__main__":
    main()
