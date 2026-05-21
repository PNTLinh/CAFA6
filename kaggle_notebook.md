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
!python /kaggle/working/CAFA6/scripts/kaggle_fix_dgl.py
```

Script tự cài **DGL CUDA** (không dùng `pip install dgl==2.1.0` CPU).

Phải in: `OK  DGL 2.1.0  GraphDataLoader OK  device=cuda:0`

**Settings:** GPU T4 bật, Internet ON.

### Cell 2 dự phòng (upload `scripts/kaggle_fix_dgl.py` nếu chưa có trên repo)

Chỉ dùng nếu không có `kaggle_fix_dgl.py` — tốt nhất copy file script từ repo.

```python
!pip uninstall -y dgl torchdata
!pip install -q dgl==2.1.0
!python /kaggle/working/CAFA6/scripts/kaggle_fix_dgl.py
```

<details><summary>Cell patch thủ công (cũ)</summary>

```python
!pip uninstall -y dgl torchdata
!pip install -q dgl==2.1.0
import importlib, site, shutil, sys
from pathlib import Path

REPL = [
    ("from torchdata.datapipes.iter import IterableWrapper, IterDataPipe",
     "from torch.utils.data import IterDataPipe\nfrom torch.utils.data.datapipes.iter import IterableWrapper"),
    ("from torchdata.datapipes.iter import IterDataPipe", "from torch.utils.data import IterDataPipe"),
    ("import torchdata.datapipes as dp", "import torch.utils.data.datapipes as dp"),
    ("import torchdata.dataloader2.graph as dp_utils", "dp_utils = None  # patched"),
    ("from torch.utils.data import functional_datapipe",
     "def functional_datapipe(name):\n    def _d(c): return c\n    return _d"),
    ("from .dataloader import *", "from .dataloader import GraphDataLoader  # patched"),
    ("from .dist_dataloader import *", "# from .dist_dataloader import *  # patched"),
    ("from ..distributed import DistGraph",
     "try:\n    from ..distributed import DistGraph\nexcept Exception:\n    class DistGraph: pass"),
]
for k in list(sys.modules):
    if k == "dgl" or k.startswith("dgl.") or k == "torchdata" or k.startswith("torchdata."):
        del sys.modules[k]

for sp in site.getsitepackages():
    root = Path(sp) / "dgl"
    if not (root / "graphbolt").is_dir():
        continue
    for py in root.rglob("*.py"):
        t = py.read_text()
        n = t
        for a, b in REPL:
            n = n.replace(a, b)
        if n != t:
            py.write_text(n)
            shutil.rmtree(py.parent / "__pycache__", ignore_errors=True)
            print("patched", py.relative_to(root))
    break

dgl = importlib.import_module("dgl")
import torch
g = dgl.graph(([0,1],[1,2]))
if torch.cuda.is_available():
    g = g.to("cuda")
print("OK", dgl.__version__, g.device)
```

</details>

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
