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


_CUDA_PROBE = """
import dgl, torch
g = dgl.graph(([0, 1], [1, 2]))
if torch.cuda.is_available():
    g.to("cuda")
print("cuda_ok")
"""


def dgl_cuda_works() -> bool:
    """True if installed libdgl supports CUDA (pip success is not enough)."""
    r = subprocess.run(
        [sys.executable, "-c", _CUDA_PROBE],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and "cuda_ok" in (r.stdout or "")


def _try_install_dgl_from(url: str) -> bool:
    pip("uninstall", "-y", "dgl")
    print(f"[install] pip install dgl==2.1.0 -f {url}")
    if pip("install", "-q", "dgl==2.1.0", "-f", url) != 0:
        return False
    if dgl_cuda_works():
        print("[install] DGL CUDA OK")
        return True
    print("[install] wheel installed but libdgl has no CUDA — try next URL")
    return False


def install_dgl_cuda() -> None:
    """Install DGL 2.1 with CUDA. Kaggle PyTorch 2.10+cu128 has no official wheel → fallback."""
    import torch

    pip("uninstall", "-y", "dgl", "torchdata")
    if not torch.cuda.is_available():
        print("[install] No CUDA in PyTorch — enable GPU T4 on Kaggle")
        pip("install", "-q", "dgl==2.1.0")
        return

    tv = ".".join(torch.__version__.split(".")[:2])
    cuda = torch.version.cuda or ""
    cu = "cu" + cuda.replace(".", "") if cuda else "cu124"

    urls = [
        f"https://data.dgl.ai/wheels/torch-{tv}/{cu}/repo.html",
        f"https://data.dgl.ai/wheels/{cu}/repo.html",
        "https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html",
        "https://data.dgl.ai/wheels/torch-2.5/cu124/repo.html",
        "https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html",
        "https://data.dgl.ai/wheels/cu124/repo.html",
        "https://data.dgl.ai/wheels/cu121/repo.html",
    ]
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        if _try_install_dgl_from(url):
            return

    # Kaggle default: torch 2.10+cu128 — DGL 2.1 has no cu128 wheel; use PyTorch 2.6+cu124 stack.
    print(
        "[install] No DGL CUDA wheel for "
        f"torch {torch.__version__} cuda {cuda}. "
        "Installing PyTorch 2.6.0+cu124 + DGL cu124 (compatible with T4)..."
    )
    pip(
        "install",
        "-q",
        "torch==2.6.0",
        "torchvision",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    )
    import importlib

    importlib.invalidate_caches()
    import torch as torch_mod

    print(f"[install] torch now {torch_mod.__version__} cuda {torch_mod.version.cuda}")

    if _try_install_dgl_from("https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html"):
        return

    raise RuntimeError(
        "Could not install DGL with CUDA. See kaggle_notebook.md or open an issue."
    )


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
