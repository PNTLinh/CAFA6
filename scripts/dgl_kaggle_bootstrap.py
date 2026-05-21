"""
Bootstrap DGL on Kaggle (Python 3.12 + CUDA 12).
- Installs torchdata (pinned) + DGL wheel matching torch CUDA
- Registers shim: torchdata.datapipes -> torch.utils.data.datapipes
  (fixes: ModuleNotFoundError: No module named 'torchdata.datapipes')

Usage in notebook BEFORE `import dgl`:
    %run /kaggle/working/CAFA6/scripts/dgl_kaggle_bootstrap.py
or:
    exec(open("scripts/dgl_kaggle_bootstrap.py").read())
"""
from __future__ import annotations

import subprocess
import sys
import types


def pip(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", *args]).returncode


def register_torchdata_shim() -> None:
    """Map legacy torchdata.datapipes imports used by DGL graphbolt."""
    import torch.utils.data.datapipes.iter as torch_iter

    torchdata_mod = types.ModuleType("torchdata")
    datapipes_mod = types.ModuleType("torchdata.datapipes")
    datapipes_mod.iter = torch_iter
    torchdata_mod.datapipes = datapipes_mod

    sys.modules["torchdata"] = torchdata_mod
    sys.modules["torchdata.datapipes"] = datapipes_mod
    sys.modules["torchdata.datapipes.iter"] = torch_iter


def install_dgl() -> None:
    import torch

    pip("install", "-q", "torchdata==0.7.1")
    pip("uninstall", "-y", "dgl")

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
            if pip("install", "-q", "dgl", "-f", url) == 0:
                break
        else:
            pip("install", "-q", "dgl==2.1.0")
    else:
        pip("install", "-q", "dgl==2.1.0")


def verify() -> None:
    import torch

    register_torchdata_shim()
    import dgl  # noqa: WPS433

    g = dgl.graph(([0, 1], [1, 2]))
    if torch.cuda.is_available():
        g = g.to("cuda")
    print(f"OK  DGL {dgl.__version__}  PyTorch {torch.__version__}  CUDA {torch.version.cuda}  device={g.device}")


if __name__ == "__main__":
    install_dgl()
    verify()
