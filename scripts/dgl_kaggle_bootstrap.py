"""
Bootstrap DGL on Kaggle (Python 3.12 + CUDA 12).

Run: !python /kaggle/working/CAFA6/scripts/dgl_kaggle_bootstrap.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def pip(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", *args]).returncode


def install_dgl() -> None:
    import torch

    pip("uninstall", "-y", "dgl", "torchdata")

    cuda = torch.version.cuda
    if cuda:
        cu = "cu" + cuda.replace(".", "")
        tv = ".".join(torch.__version__.split(".")[:2])
        urls = [
            f"https://data.dgl.ai/wheels/torch-{tv}/{cu}/repo.html",
            f"https://data.dgl.ai/wheels/{cu}/repo.html",
            "https://data.dgl.ai/wheels/cu124/repo.html",
        ]
        for url in urls:
            if pip("install", "-q", "dgl==2.1.0", "-f", url) == 0:
                break
        else:
            pip("install", "-q", "dgl==2.1.0")
    else:
        pip("install", "-q", "dgl==2.1.0")


def patch_graphbolt_on_disk() -> None:
    """Patch DGL graphbolt/base.py before first import (no import dgl here)."""
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
            print("Patched", base_py)
        elif new in text:
            print("Already patched", base_py)
        else:
            print("WARN: unexpected graphbolt/base.py at", base_py)
        return
    print("WARN: dgl graphbolt/base.py not found in site-packages")


def verify() -> None:
    import torch

    from model.dgl_compat import apply_dgl_compat

    patch_graphbolt_on_disk()
    apply_dgl_compat()

    for key in list(sys.modules):
        if key == "dgl" or key.startswith("dgl."):
            del sys.modules[key]

    import dgl

    g = dgl.graph(([0, 1], [1, 2]))
    if torch.cuda.is_available():
        g = g.to("cuda")
    print(
        f"OK  DGL {dgl.__version__}  PyTorch {torch.__version__}  "
        f"CUDA {torch.version.cuda}  device={g.device}"
    )


if __name__ == "__main__":
    install_dgl()
    verify()
