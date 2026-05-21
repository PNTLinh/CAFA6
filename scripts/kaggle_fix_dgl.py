#!/usr/bin/env python3
"""Install DGL 2.1 (CUDA) + patch torchdata on Kaggle Py3.12.

Run: python scripts/kaggle_fix_dgl.py
Skip pip install: python scripts/kaggle_fix_dgl.py --no-install
"""
from __future__ import annotations

import importlib
import re
import shutil
import site
import subprocess
import sys
from pathlib import Path


def pip(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", *args]).returncode


def install_dgl_cuda() -> None:
    """Install DGL 2.1 wheel with CUDA matching current PyTorch."""
    import torch

    pip("uninstall", "-y", "dgl", "torchdata")
    cuda = torch.version.cuda
    if not cuda:
        print("[install] No CUDA in PyTorch — installing CPU dgl (enable GPU on Kaggle!)")
        pip("install", "-q", "dgl==2.1.0")
        return

    cu = "cu" + cuda.replace(".", "")
    tv = ".".join(torch.__version__.split(".")[:2])
    urls = [
        f"https://data.dgl.ai/wheels/torch-{tv}/{cu}/repo.html",
        f"https://data.dgl.ai/wheels/{cu}/repo.html",
        "https://data.dgl.ai/wheels/cu124/repo.html",
        "https://data.dgl.ai/wheels/cu121/repo.html",
    ]
    for url in urls:
        print(f"[install] pip install dgl==2.1.0 -f {url}")
        if pip("install", "-q", "dgl==2.1.0", "-f", url) == 0:
            return
    print("[install] WARN: CUDA wheel failed, falling back to CPU dgl")
    pip("install", "-q", "dgl==2.1.0")


def dgl_root() -> Path:
    for sp in site.getsitepackages():
        p = Path(sp) / "dgl"
        if (p / "__init__.py").is_file():
            return p
    raise RuntimeError("dgl not installed — run: pip install dgl==2.1.0")


def patch_file(py: Path) -> bool:
    text = py.read_text(encoding="utf-8")
    orig = text

    # --- torchdata -> torch (any file) ---
    text = text.replace(
        "from torchdata.datapipes.iter import IterableWrapper, IterDataPipe",
        "from torch.utils.data import IterDataPipe\n"
        "from torch.utils.data.datapipes.iter import IterableWrapper",
    )
    text = text.replace(
        "from torchdata.datapipes.iter import IterDataPipe, IterableWrapper",
        "from torch.utils.data import IterDataPipe\n"
        "from torch.utils.data.datapipes.iter import IterableWrapper",
    )
    text = text.replace(
        "from torchdata.datapipes.iter import IterDataPipe",
        "from torch.utils.data import IterDataPipe",
    )
    text = text.replace(
        "from torchdata.datapipes.iter import Mapper",
        "from torch.utils.data.datapipes.iter import Mapper",
    )
    text = text.replace("import torchdata.datapipes as dp", "import torch.utils.data.datapipes as dp")
    text = text.replace(
        "import torchdata.dataloader2.graph as dp_utils",
        "dp_utils = None  # kaggle_fix_dgl",
    )

    # graphbolt/base.py — no re-register on re-import
    if py.name == "base.py" and "graphbolt" in py.parts:
        text = text.replace(
            "from torch.utils.data import functional_datapipe",
            "def functional_datapipe(name):\n"
            "    def _decorator(cls):\n        return cls\n    return _decorator",
        )

    # dataloading/dataloader.py — optional distributed
    if py.name == "dataloader.py" and "dataloading" in py.parts:
        if "from ..distributed import DistGraph" in text and "kaggle_fix_dgl" not in text:
            text = text.replace(
                "from ..distributed import DistGraph",
                "try:\n    from ..distributed import DistGraph\n"
                "except Exception:\n    class DistGraph:\n        pass  # kaggle_fix_dgl",
            )

    if text == orig:
        return False
    py.write_text(text, encoding="utf-8")
    shutil.rmtree(py.parent / "__pycache__", ignore_errors=True)
    return True


def fix_dataloading_init(root: Path) -> None:
    py = root / "dataloading" / "__init__.py"
    text = py.read_text(encoding="utf-8")
    new_block = '''if F.get_preferred_backend() == "pytorch":
    from .spot_target import *
    from .dataloader import GraphDataLoader  # kaggle_fix_dgl
'''
    text = re.sub(
        r"if F\.get_preferred_backend\(\) == \"pytorch\":.*?(?=\n(?:if |\Z))",
        new_block,
        text,
        count=1,
        flags=re.DOTALL,
    )
    py.write_text(text, encoding="utf-8")
    shutil.rmtree(py.parent / "__pycache__", ignore_errors=True)
    print(f"[fix] {py}")


def fix_graphbolt_init(root: Path) -> None:
    py = root / "graphbolt" / "__init__.py"
    text = py.read_text(encoding="utf-8")
    text = text.replace("from .dataloader import *", "# from .dataloader import *  # kaggle_fix_dgl")
    if "load_graphbolt()" in text and "kaggle_fix_dgl" not in text:
        text = text.replace(
            "load_graphbolt()",
            "try:\n    load_graphbolt()\nexcept Exception as _e:\n"
            "    import warnings\n    warnings.warn(f'graphbolt C++ skipped: {_e}')",
        )
    py.write_text(text, encoding="utf-8")
    shutil.rmtree(py.parent / "__pycache__", ignore_errors=True)
    print(f"[fix] {py}")


def fix_dgl_init(root: Path) -> None:
    py = root / "__init__.py"
    text = py.read_text(encoding="utf-8")
    old = 'if backend_name == "pytorch":\n    from . import distributed'
    new = (
        'if backend_name == "pytorch":\n'
        "    try:\n        from . import distributed  # kaggle_fix_dgl\n"
        "    except Exception:\n        pass"
    )
    if old in text:
        text = text.replace(old, new)
        py.write_text(text, encoding="utf-8")
        print(f"[fix] {py}")


def main() -> None:
    if "--no-install" not in sys.argv:
        install_dgl_cuda()

    root = dgl_root()
    n = 0
    for py in root.rglob("*.py"):
        if patch_file(py):
            print(f"[fix] {py.relative_to(root.parent)}")
            n += 1
    fix_dataloading_init(root)
    fix_graphbolt_init(root)
    fix_dgl_init(root)

    for key in list(sys.modules):
        if key == "dgl" or key.startswith("dgl."):
            del sys.modules[key]
    importlib.invalidate_caches()

    dgl = importlib.import_module("dgl")
    from dgl.dataloading import GraphDataLoader

    import torch

    g = dgl.graph(([0, 1], [1, 2]))
    device_msg = "cpu"
    if torch.cuda.is_available():
        try:
            g = g.to("cuda")
            device_msg = str(g.device)
        except Exception as exc:
            device_msg = f"cpu-only DGL build ({exc})"
            print(
                "[WARN] DGL imported but CUDA failed. Re-run this script without prior "
                "'pip install dgl==2.1.0' (CPU). Enable GPU T4 on Kaggle."
            )
    else:
        print("[WARN] torch.cuda.is_available() is False — turn on GPU in notebook settings.")

    print(
        f"OK  DGL {dgl.__version__}  GraphDataLoader OK  device={device_msg}  "
        f"({n} files patched)"
    )
    if "cpu" in device_msg and torch.cuda.is_available():
        sys.exit(1)


if __name__ == "__main__":
    main()
