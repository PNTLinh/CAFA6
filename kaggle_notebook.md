# Kaggle notebook — copy từng cell

Bật **GPU T4** + **Internet**. Nếu lỗi DGL/numpy → **Restart session** → chạy lại từ Cell 1.

Gắn dataset `cafa6-data` (từ `pack_for_kaggle.py`) vào notebook.

---

## Cell 1 — Clone repo

```python
%cd /kaggle/working
!rm -rf CAFA6
!git clone https://github.com/PNTLinh/CAFA6.git
%cd CAFA6
```

*(Đã clone rồi, chỉ cập nhật code: `%cd /kaggle/working/CAFA6` rồi `!git pull`)*

---

## Cell 2 — Cài thư viện + DGL CUDA

```python
!pip install -q "numpy>=1.26,<2.4" "scipy>=1.11,<1.16"
!pip install -q packaging fair-esm transformers biopython tqdm scikit-learn pandas networkx requests psutil
!python /kaggle/working/CAFA6/scripts/kaggle_fix_dgl.py
```

---

## Cell 3 — Kiểm tra GPU

```python
import torch
import dgl

g = dgl.graph(([0, 1], [1, 2]))
if torch.cuda.is_available():
    g = g.to("cuda")
print("OK", torch.__version__, dgl.__version__, g.device)
```

---

## Cell 4 — Nối dữ liệu

```python
!python /kaggle/working/CAFA6/scripts/kaggle_link_data.py
!ls /kaggle/working/CAFA6/proceed_data/ppi_graph_global
!ls /kaggle/working/CAFA6/divided_data/*_train_dataset
```

## Cell 4.5 — Khóa `DATA_DIR` và kiểm tra split

Chạy cell này trước khi train/eval để tránh notebook cũ hoặc kernel cũ quay về `D:/CAFA6`.

```python
import os
from pathlib import Path

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
print("DATA_DIR =", os.environ["DATA_DIR"])

data_root = Path(os.environ["DATA_DIR"])
for rel_path in ["proceed_data", "divided_data"]:
    print(rel_path, (data_root / rel_path).exists())

for rel_path in [
    "divided_data/cc_test_dataset",
    "divided_data/cc_valid_dataset",
    "divided_data/mf_test_dataset",
    "divided_data/mf_valid_dataset",
    "divided_data/bp_test_dataset",
    "divided_data/bp_valid_dataset",
]:
    print(rel_path, (data_root / rel_path).exists())
```

Nếu `cc_test_dataset` và `cc_valid_dataset` đều `False`, dữ liệu Kaggle chưa đủ để eval branch `cc`; cần re-run `kaggle_link_data.py` hoặc pack lại dataset.

## Cell 4.6b — Vá `mf_train_dataset` nếu bị EOFError

Chạy cell này chỉ khi train nhánh `mf` và file `mf_train_dataset` bị rỗng/hỏng. Cell sẽ tìm file `mf_train_dataset` trong `/kaggle/input`, copy bản hợp lệ sang thư mục làm việc, rồi kiểm tra lại kích thước file.

```python
from pathlib import Path
import shutil

work_file = Path("/kaggle/working/CAFA6/divided_data/mf_train_dataset")
if work_file.exists() and work_file.stat().st_size > 0:
    print("mf_train_dataset OK:", work_file)
else:
    candidates = [p for p in Path("/kaggle/input").rglob("mf_train_dataset") if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy mf_train_dataset hợp lệ trong /kaggle/input. Hãy gắn dataset mf-train1 rồi chạy lại cell này."
        )

    source = max(candidates, key=lambda p: p.stat().st_size)
    work_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, work_file)
    print(f"copied {source} -> {work_file}")
    print("size(bytes)=", work_file.stat().st_size)
```

## Cell 4.6 — Sửa lại `eval_Struct2GO2.py` nếu notebook đang dùng bản cũ

Chạy cell này nếu bạn thấy traceback vẫn trỏ tới `D:/CAFA6` hoặc line 248 cũ sau khi clone repo vào Kaggle.

```python
from pathlib import Path

eval_path = Path("/kaggle/working/CAFA6/eval_Struct2GO2.py")
text = eval_path.read_text(encoding="utf-8")
old = '    data_dir = os.environ.get("DATA_DIR", "D:/CAFA6")\n'
new = '    data_dir = _resolve_data_dir()\n'

if old in text and new not in text:
    text = text.replace(old, new)
    if 'def _resolve_data_dir() -> str:' not in text:
        anchor = '_ACS_FILES = {\n    "mf": "human_MF_ACS.json",\n    "cc": "human_CC_ACS.json",\n    "bp": "human_BP_ACS.json",\n}\n\n'
        insert = anchor + '''
def _resolve_data_dir() -> str:
    """Pick the first usable CAFA6 data root for local or Kaggle runs."""
    candidates = []
    env_data_dir = os.environ.get("DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir))
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())
    candidates.append(Path("D:/CAFA6"))

    for candidate in candidates:
        if (candidate / "divided_data").exists() and (candidate / "proceed_data").exists():
            return str(candidate)

    return env_data_dir or str(Path(__file__).resolve().parent)

'''
        text = text.replace(anchor, insert, 1)
    eval_path.write_text(text, encoding="utf-8")
    print("patched stale eval_Struct2GO2.py")
else:
    print("eval_Struct2GO2.py already current")
```

---

## Cell 5 — Train CC → MF → BP, lưu từng nhánh, cập nhật zip

~1–2 giờ tổng (BP chậm nhất). Sau mỗi nhánh zip được ghi lại tại `/kaggle/working/cafa6_output.zip`.

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py
```

**MF đã train xong** (chỉ chạy cc + bp, vẫn copy mf cũ vào zip nếu có):

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py --branches cc mf bp --skip mf
```

**OOM** → thêm `-batch_size 64`:

```python
!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py -batch_size 64
```

**Chỉ train, không eval** (nhanh hơn):

```python
!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py --no-eval
```

**Eval riêng nhánh CC** (nếu muốn chạy trực tiếp thay vì qua `kaggle_run_branches.py`):

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/eval_Struct2GO2.py -branch cc --no-baseline-parity --split auto
```

---

## Cell 6 — Lưu log + model + test result + zip (copy-paste)

Chạy **sau train/eval** (mỗi nhánh hoặc cả 3). Output nằm dưới `/kaggle/working/` → tab **Output** của notebook.

```python
import os
from pathlib import Path

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"

# Đổi danh sách nhánh nếu chỉ mới xong 1–2 nhánh: ví dụ ["cc"] hoặc ["cc", "mf"]
BRANCHES = ["cc", "mf", "bp"]
ZIP_NAME = "cafa6_output.zip"

branch_args = " ".join(BRANCHES)
!python /kaggle/working/CAFA6/scripts/kaggle_save_results.py --branches {branch_args} --zip --zip-name {ZIP_NAME}
```

---

## Cell 7 — Kiểm tra đã lưu đủ chưa

```python
from pathlib import Path

zip_path = Path("/kaggle/working/cafa6_output.zip")
print("zip:", zip_path, f"({zip_path.stat().st_size / 1e6:.1f} MB)" if zip_path.is_file() else "MISSING")

for folder in ["log", "save_models", "test_result"]:
    p = Path("/kaggle/working") / folder
    print(f"\n=== {folder}/ ===")
    if not p.is_dir():
        print("  (trống)")
        continue
    for f in sorted(p.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            unit = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size} B"
            print(f"  {f.name}  ({unit})")

# Metric test (nếu đã eval)
import re
for log in sorted(Path("/kaggle/working/log").glob("test_*.log")):
    text = log.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"f_score\s+([\d.]+).*?auc\s+([\d.]+).*?aupr\s+([\d.]+)", text, re.S)
    if hits:
        f, a, u = hits[-1]
        print(f"\n{log.name}: F-max={f}, AUC={a}, AUPR={u}")
```

---

## Cell 8 — Tải zip về máy

```python
from IPython.display import FileLink, display

display(FileLink("/kaggle/working/cafa6_output.zip"))
```

Bấm link **cafa6_output.zip** trong output, hoặc **Save Version → Save & Run All** rồi tải từ tab **Output** bên phải.

**Nội dung zip:**

```
log/mf.log  log/cc.log  log/bp.log
log/test_mf.log  log/test_cc.log  log/test_bp.log
save_models/bestmodel_*.pkl  final_*.pkl
test_result/*_result.json  *_pred_actual.pkl  *_roc_curve.png
```
