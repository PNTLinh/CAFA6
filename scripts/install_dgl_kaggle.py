"""
Install DGL with CUDA matching the current PyTorch (for Kaggle / Linux GPU).
Run in notebook:  !python scripts/install_dgl_kaggle.py
"""
from __future__ import annotations

import subprocess
import sys


def pip(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True)


def register_torchdata_shim() -> None:
    import types

    import torch.utils.data.datapipes.iter as torch_iter

    torchdata_mod = types.ModuleType("torchdata")
    datapipes_mod = types.ModuleType("torchdata.datapipes")
    datapipes_mod.iter = torch_iter
    torchdata_mod.datapipes = datapipes_mod
    sys.modules["torchdata"] = torchdata_mod
    sys.modules["torchdata.datapipes"] = datapipes_mod
    sys.modules["torchdata.datapipes.iter"] = torch_iter


def try_import_cuda() -> bool:
    register_torchdata_shim()
    import dgl
    import torch

    g = dgl.graph(([0, 1], [1, 2]))
    if torch.cuda.is_available():
        g = g.to("cuda")
    print(f"DGL {dgl.__version__}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}, graph device={g.device}")
    return True


def main() -> None:
    import torch

    pip("install", "-q", "torchdata==0.7.1")

    pip("uninstall", "-y", "dgl")

    cuda = torch.version.cuda
    if not cuda:
        print("No CUDA in PyTorch — installing CPU DGL")
        pip("install", "-q", "dgl==2.1.0")
        try_import_cuda()
        return

    cu_tag = "cu" + cuda.replace(".", "")  # 12.4 -> cu124
    torch_mm = ".".join(torch.__version__.split(".")[:2])  # 2.5.1 -> 2.5

    candidates = [
        f"https://data.dgl.ai/wheels/torch-{torch_mm}/{cu_tag}/repo.html",
        f"https://data.dgl.ai/wheels/{cu_tag}/repo.html",
        "https://data.dgl.ai/wheels/cu121/repo.html",
        "https://data.dgl.ai/wheels/cu118/repo.html",
    ]

    for url in candidates:
        print(f"Trying: pip install dgl -f {url}")
        r = pip("install", "-q", "dgl", "-f", url)
        if r.returncode != 0:
            print(r.stderr or r.stdout)
            pip("uninstall", "-y", "dgl")
            continue
        try:
            try_import_cuda()
            print("SUCCESS")
            return
        except OSError as e:
            print(f"Import failed: {e}")
            pip("uninstall", "-y", "dgl")

    raise SystemExit(
        "Could not install CUDA DGL. In notebook run:\n"
        "  import torch; print(torch.__version__, torch.version.cuda)\n"
        "Then open https://www.dgl.ai/pages/start.html and pick matching wheel."
    )


if __name__ == "__main__":
    main()
