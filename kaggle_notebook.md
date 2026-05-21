# Kaggle — CAFA6

**Restart session** nếu từng chạy shim `torchdata` cũ.

---

## Cell 1 — Clone

```python
%cd /kaggle/working
!rm -rf CAFA6
!git clone https://github.com/PNTLinh/CAFA6.git
%cd CAFA6
```

Nếu GitHub chưa cập nhật: upload/copy `model/dgl_patch.py` mới (có patch `graphbolt/__init__.py`).

---

## Cell 2 — DGL (copy nguyên khối)

```python
!pip install -q packaging fair-esm transformers biopython tqdm
!pip uninstall -y dgl
!pip install -q torchdata==0.11.0
!pip install -q dgl==2.1.0
!python /kaggle/working/CAFA6/model/dgl_patch.py
```

Phải thấy **2 dòng** patch:

```
[dgl_patch] .../dgl/graphbolt/base.py
[dgl_patch] .../dgl/graphbolt/__init__.py
[dgl_patch] OK  DGL 2.1.0 ...
```

### Cell 2 dự phòng (không cần file repo)

```python
!pip install -q torchdata==0.11.0 dgl==2.1.0
import importlib, site, shutil, sys
from pathlib import Path

def purge(p):
    for k in list(sys.modules):
        if k == p or k.startswith(p + "."):
            del sys.modules[k]

purge("torchdata"); purge("dgl")
td = importlib.import_module("torchdata")
assert getattr(td, "__path__", None), "torchdata phải là package thật, không phải shim"
importlib.import_module("torchdata.dataloader2.graph")

for sp in site.getsitepackages():
    root = Path(sp) / "dgl"
    if not (root / "graphbolt/base.py").exists():
        continue
    (root / "graphbolt/base.py").write_text(
        (root / "graphbolt/base.py").read_text().replace(
            "from torchdata.datapipes.iter import IterDataPipe",
            "from torch.utils.data import IterDataPipe",
        )
    )
    init = root / "graphbolt/__init__.py"
    init.write_text(
        init.read_text().replace(
            "from .dataloader import *",
            "# from .dataloader import *  # patched",
        )
    )
    shutil.rmtree(root / "graphbolt/__pycache__", ignore_errors=True)
    print("patched", root)
    break

purge("dgl")
dgl = importlib.import_module("dgl")
import torch
g = dgl.graph(([0,1],[1,2]))
if torch.cuda.is_available():
    g = g.to("cuda")
print("OK", dgl.__version__, g.device)
```

---

## Cell 3 — Data

```python
!python /kaggle/working/CAFA6/scripts/kaggle_link_data.py
```

---

## Cell 4 — Train

```python
import os
os.environ["DGL_CUDA"] = "1"
os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
!cd /kaggle/working/CAFA6 && python train_Struct2GO2.py -branch mf --kaggle
```
