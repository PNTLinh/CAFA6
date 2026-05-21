"""Kaggle / Py3.12: DGL graphbolt imports torchdata.datapipes (removed in new torchdata)."""
from __future__ import annotations

import sys
import types


def apply_dgl_compat() -> None:
    if "torchdata.datapipes.iter" in sys.modules:
        return
    try:
        import torch.utils.data.datapipes.iter as torch_iter
    except ImportError:
        return

    torchdata_mod = types.ModuleType("torchdata")
    datapipes_mod = types.ModuleType("torchdata.datapipes")
    datapipes_mod.iter = torch_iter
    torchdata_mod.datapipes = datapipes_mod

    sys.modules["torchdata"] = torchdata_mod
    sys.modules["torchdata.datapipes"] = datapipes_mod
    sys.modules["torchdata.datapipes.iter"] = torch_iter


apply_dgl_compat()
