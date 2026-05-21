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


def _try_install_dgl_from(url: str, *, force: bool = False) -> bool:
    pip("uninstall", "-y", "dgl")
    print(f"[install] pip install dgl==2.1.0 -f {url}")
    args = ["install", "-q", "dgl==2.1.0", "-f", url]
    if force:
        args = ["install", "-q", "--force-reinstall", "dgl==2.1.0", "-f", url]
    if pip(*args) != 0:
        return False
    if dgl_cuda_works():
        print("[install] DGL CUDA OK")
        return True
    print("[install] wheel installed but libdgl has no CUDA — try next URL")
    return False


def _install_pytorch_26_cu124() -> None:
    """Align torch/torchaudio/torchvision — pip cannot downgrade torch alone on Kaggle."""
    print("[install] PyTorch 2.6.0+cu124 stack (torchaudio 2.10 blocks torch-only downgrade)")
    pip("uninstall", "-y", "torch", "torchaudio", "torchvision")
    pip(
        "install",
        "-q",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    )
    import importlib

    importlib.invalidate_caches()
    import torch as torch_mod

    ver = torch_mod.__version__
    print(f"[install] torch now {ver} cuda {torch_mod.version.cuda}")
    if not ver.startswith("2.6"):
        raise RuntimeError(f"torch downgrade failed (still {ver}); restart session and rerun.")


def install_dgl_cuda() -> None:
    """Install DGL 2.1 with CUDA. Kaggle torch 2.10+cu128: prefer cu121 DGL or full torch 2.6 stack."""
    import torch

    pip("uninstall", "-y", "torchdata")
    if not torch.cuda.is_available():
        print("[install] No CUDA in PyTorch — enable GPU T4 on Kaggle")
        pip("install", "-q", "dgl==2.1.0")
        return

    if dgl_cuda_works():
        print("[install] existing DGL already supports CUDA — skip pip")
        return

    tv = ".".join(torch.__version__.split(".")[:2])
    cuda = torch.version.cuda or ""

    # cu121 wheel often works on Kaggle T4 (CUDA 12.x driver) even with torch 2.10+cu128.
    urls = [
        "https://data.dgl.ai/wheels/cu121/repo.html",
        "https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html",
        "https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html",
        "https://data.dgl.ai/wheels/torch-2.5/cu124/repo.html",
        "https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html",
        "https://data.dgl.ai/wheels/cu124/repo.html",
        f"https://data.dgl.ai/wheels/torch-{tv}/cu121/repo.html",
        f"https://data.dgl.ai/wheels/torch-{tv}/cu124/repo.html",
    ]
    if cuda:
        cu = "cu" + cuda.replace(".", "")
        urls.insert(0, f"https://data.dgl.ai/wheels/{cu}/repo.html")
        urls.insert(0, f"https://data.dgl.ai/wheels/torch-{tv}/{cu}/repo.html")

    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        if _try_install_dgl_from(url):
            return

    print(
        f"[install] No DGL CUDA wheel matched torch {torch.__version__} cuda {cuda}. "
        "Switching to PyTorch 2.6.0+cu124 + DGL cu124..."
    )
    _install_pytorch_26_cu124()
    for url in (
        "https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html",
        "https://data.dgl.ai/wheels/cu124/repo.html",
        "https://data.dgl.ai/wheels/cu121/repo.html",
    ):
        if _try_install_dgl_from(url, force=True):
            return

    raise RuntimeError(
        "Could not install DGL with CUDA. Restart session, then rerun this script."
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
