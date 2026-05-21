"""
Fix DGL 2.1 on Kaggle Py3.12 / PyTorch 2.6.

Patches installed dgl/*.py on disk (graphbolt imports torchdata) and registers a
small torchdata shim. Call ensure_dgl_importable() before `import dgl`.
"""
from __future__ import annotations

import shutil
import site
import sys
from pathlib import Path

_REPLACEMENTS = (
    ("from torchdata.datapipes.iter import IterDataPipe", "from torch.utils.data import IterDataPipe"),
    ("from torchdata.datapipes import iter as torch_iter", "from torch.utils.data import IterDataPipe  # patched"),
)


def _dgl_roots() -> list[Path]:
    roots: list[Path] = []
    for sp in site.getsitepackages():
        root = Path(sp) / "dgl"
        if (root / "__init__.py").is_file():
            roots.append(root)
    try:
        import importlib.util

        spec = importlib.util.find_spec("dgl")
        if spec and spec.origin:
            root = Path(spec.origin).resolve().parent
            if root not in roots:
                roots.append(root)
    except Exception:
        pass
    return roots


def patch_dgl_on_disk(verbose: bool = True) -> int:
    """Rewrite torchdata imports in every file under site-packages/dgl. Returns patch count."""
    n = 0
    for root in _dgl_roots():
        for py in root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            new = text
            for old, repl in _REPLACEMENTS:
                new = new.replace(old, repl)
            if new == text:
                continue
            py.write_text(new, encoding="utf-8")
            cache = py.with_suffix(".pyc")
            if cache.exists():
                cache.unlink()
            pycache = py.parent / "__pycache__"
            if pycache.is_dir():
                shutil.rmtree(pycache, ignore_errors=True)
            n += 1
            if verbose:
                print(f"[dgl_patch] {py}")
    return n


def _register_torchdata_shim() -> None:
    import types

    from torch.utils.data import IterDataPipe

    for key in list(sys.modules):
        if key == "torchdata" or key.startswith("torchdata."):
            del sys.modules[key]

    iter_mod = types.ModuleType("torchdata.datapipes.iter")
    iter_mod.IterDataPipe = IterDataPipe
    datapipes_mod = types.ModuleType("torchdata.datapipes")
    datapipes_mod.iter = iter_mod
    torchdata_mod = types.ModuleType("torchdata")
    torchdata_mod.datapipes = datapipes_mod
    sys.modules["torchdata"] = torchdata_mod
    sys.modules["torchdata.datapipes"] = datapipes_mod
    sys.modules["torchdata.datapipes.iter"] = iter_mod


def _purge_dgl_modules() -> None:
    for key in list(sys.modules):
        if key == "dgl" or key.startswith("dgl."):
            del sys.modules[key]


def ensure_dgl_importable(verbose: bool = True) -> None:
    """Patch DGL install + shim; safe to call multiple times."""
    patched = patch_dgl_on_disk(verbose=verbose)
    if verbose and patched == 0:
        print("[dgl_patch] no text changes (already patched or dgl not installed yet)")
    _register_torchdata_shim()
    _purge_dgl_modules()
    import dgl  # noqa: WPS433

    if verbose:
        import torch

        print(f"[dgl_patch] OK  DGL {dgl.__version__}  torch {torch.__version__}")


def main() -> None:
    ensure_dgl_importable(verbose=True)


if __name__ == "__main__":
    main()
