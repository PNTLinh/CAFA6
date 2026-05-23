# Kaggle — CAFA6

Notebook này dành cho **Kaggle T4** và ưu tiên dùng script có sẵn trong repo để tránh lỗi thư viện `torchdata`/`dgl`.

## Trước khi chạy

- Bật **GPU T4** trong Kaggle Notebook.
- Bật **Internet**.
- Nếu trước đó bạn đã thử cài `torchdata` hoặc `dgl` thủ công, hãy **Restart session** trước khi chạy lại notebook này.

## Cell 1 — Clone repo

```python
%cd /kaggle/working
!rm -rf CAFA6
!git clone https://github.com/PNTLinh/CAFA6.git
%cd CAFA6
```

## Cell 2 — Cài dependencies và DGL

**Nếu lỗi numpy/scipy hoặc `torch 2.10` / DGL CUDA:** **Restart session** → chạy lại từ Cell 1.

```python
# Ghim numpy trước (tránh lỗi scipy sau khi pip)
!pip install -q "numpy>=1.26,<2.4" "scipy>=1.11,<1.16"
!pip install -q packaging fair-esm transformers biopython tqdm scikit-learn pandas networkx requests psutil

# Cài DGL CUDA (torch 2.6 + dgl 2.5.0+cu124) — KHÔNG chạy train trong cell này
!python /kaggle/working/CAFA6/scripts/kaggle_fix_dgl.py
```

Không dùng `pip install dgl` thủ công (thường ra bản CPU).

## Cell 3 — Kiểm tra CUDA + DGL

```python
import torch
import dgl

g = dgl.graph(([0, 1], [1, 2]))
if torch.cuda.is_available():
    g = g.to("cuda")
print("OK", torch.__version__, dgl.__version__, g.device)
```

Phải thấy `2.6.0+cu124`, `2.5.0+cu124` (hoặc tương đương CUDA), `device=cuda:0`.

Phải thấy thiết bị dạng `cuda:0`.

## Cell 4 — Nối dữ liệu Kaggle

Upload dataset đã pack bằng `pack_for_kaggle.py`, sau đó chạy:

```python
!python /kaggle/working/CAFA6/scripts/kaggle_link_data.py
```

Script này sẽ tự tìm `proceed_data/ppi_graph_global` và `divided_data/*` trong `/kaggle/input` rồi symlink vào `/kaggle/working/CAFA6`.

## Cell 5 — Train (~30 phút / nhánh)

Preset `--kaggle`: **mf/cc** 5 epoch, **bp** 4 epoch, `hid_dim=256`, validate 1 lần ở epoch cuối (~30 phút/nhánh trên T4).

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

import __main__
from data_processing.divide_data import MyDataSet
__main__.MyDataSet = MyDataSet

%cd /kaggle/working/CAFA6
!python train_Struct2GO2.py -branch mf --kaggle
!python train_Struct2GO2.py -branch cc --kaggle
!python train_Struct2GO2.py -branch bp --kaggle -dropout 0.1
```

Nếu **mf đã train xong** (log có `saved final` hoặc `best_fscore`), chỉ train cc + bp:

```python
# Bỏ qua dòng train mf ở trên, chỉ chạy:
!python train_Struct2GO2.py -branch cc --kaggle
!python train_Struct2GO2.py -branch bp --kaggle -dropout 0.1
```

OOM → giảm batch: `-batch_size 64`.

## Cell 6 — Lưu log + model ra Output

Chỉ **cc** và **bp** (mf đã có sẵn):

```python
!python /kaggle/working/CAFA6/scripts/kaggle_save_results.py --branches cc bp
```

Hoặc copy tay (nếu chưa có script):

```python
!mkdir -p /kaggle/working/log /kaggle/working/save_models
!cp -v /kaggle/working/CAFA6/log/cc.log /kaggle/working/CAFA6/log/bp.log /kaggle/working/log/
!cp -v /kaggle/working/CAFA6/save_models/*cc*.pkl /kaggle/working/CAFA6/save_models/*bp*.pkl /kaggle/working/save_models/ 2>/dev/null || true
!find /kaggle/working -name "*cc*.pkl" -o -name "*bp*.pkl" 2>/dev/null
!ls -lh /kaggle/working/log/ /kaggle/working/save_models/
```

Lưu cả 3 nhánh:

```python
!python /kaggle/working/CAFA6/scripts/kaggle_save_results.py --branches mf cc bp
```

Kỳ vọng:

```
/kaggle/working/log/mf.log  cc.log  bp.log
/kaggle/working/save_models/bestmodel_*.pkl  final_*.pkl
```

Zip download:

```python
!cd /kaggle/working && zip -r cafa6_output.zip log save_models
!ls -lh /kaggle/working/cafa6_output.zip
```

## Ghi chú nhanh

- Preset `--kaggle` ~30p/nhánh: `amp`, cache PPI, ít ngưỡng F-score (9).
- `-epochs`, `-batch_size`, … trên CLI vẫn được giữ nếu bạn truyền tay.
- `DATA_DIR=/kaggle/working/CAFA6` — checkpoint luôn lưu vào `{DATA_DIR}/save_models/`.
- Thứ tự: clone → install_dgl_kaggle.py → verify → kaggle_link_data.py → train → kaggle_save_results.py.
