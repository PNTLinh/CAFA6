"""Kaggle / Py3.12: DGL 2.1 graphbolt imports torchdata.datapipes.iter.IterDataPipe."""
from __future__ import annotations

import sys
import types
from pathlib import Path


def _iter_data_pipe_class():
    """IterDataPipe from PyTorch (torchdata is broken / missing on Kaggle Py3.12)."""
    for path in ("torch.utils.data", "torch.utils.data.datapipes.datapipe"):
        try:
            mod = __import__(path, fromlist=["IterDataPipe"])
            cls = getattr(mod, "IterDataPipe", None)
            if cls is not None:
                return cls
        except (ImportError, AttributeError):
            continue
    return None


def _purge_torchdata_modules() -> None:
    for key in list(sys.modules):
        if key == "torchdata" or key.startswith("torchdata."):
            del sys.modules[key]


def _purge_dgl_modules() -> None:
    for key in list(sys.modules):
        if key == "dgl" or key.startswith("dgl."):
            del sys.modules[key]


def apply_dgl_compat() -> None:
    """Register fake torchdata.datapipes.iter exposing IterDataPipe (never alias torch iter)."""
    cls = _iter_data_pipe_class()
    if cls is None:
        raise ImportError("Could not locate torch.utils.data.IterDataPipe")

    _purge_torchdata_modules()

    iter_key = "torchdata.datapipes.iter"
    iter_mod = types.ModuleType(iter_key)
    iter_mod.IterDataPipe = cls

    datapipes_mod = types.ModuleType("torchdata.datapipes")
    datapipes_mod.iter = iter_mod

    torchdata_mod = types.ModuleType("torchdata")
    torchdata_mod.datapipes = datapipes_mod

    sys.modules["torchdata"] = torchdata_mod
    sys.modules["torchdata.datapipes"] = datapipes_mod
    sys.modules[iter_key] = iter_mod

    from torchdata.datapipes.iter import IterDataPipe  # noqa: F401

    if IterDataPipe is not cls:
        raise ImportError("torchdata.datapipes.iter shim did not register IterDataPipe")


def patch_dgl_graphbolt_source() -> bool:
    """Patch installed DGL 2.1 graphbolt/base.py on disk (safe before import dgl)."""
    import site

    old = "from torchdata.datapipes.iter import IterDataPipe"
    new = "from torch.utils.data import IterDataPipe"
    for sp in site.getsitepackages():
        base_py = Path(sp) / "dgl" / "graphbolt" / "base.py"
        if not base_py.is_file():
            continue
        text = base_py.read_text(encoding="utf-8")
        if old in text:
            base_py.write_text(text.replace(old, new), encoding="utf-8")
        _purge_dgl_modules()
        return True
    return False


# Auto-apply on import (train scripts import this module first).
apply_dgl_compat()
