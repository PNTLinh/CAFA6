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

```python
!pip install -q packaging fair-esm transformers biopython tqdm scikit-learn pandas scipy networkx requests psutil
!python /kaggle/working/CAFA6/scripts/install_dgl_kaggle.py
```

Nếu cell này báo lỗi hoặc torch/dgl bị đổi phiên bản ngoài ý muốn, restart session rồi chạy lại từ đầu.

## Cell 3 — Kiểm tra CUDA + DGL

```python
import torch
from model.dgl_patch import ensure_dgl_importable

ensure_dgl_importable(verbose=False)

import dgl

g = dgl.graph(([0, 1], [1, 2]))
if torch.cuda.is_available():
    g = g.to("cuda")
print("OK", torch.__version__, dgl.__version__, g.device)
```

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

```python
!python /kaggle/working/CAFA6/scripts/kaggle_save_results.py --branches mf cc bp --skip-train-if-ok
```

Hoặc chỉ lưu mf (không train lại):

```python
!python /kaggle/working/CAFA6/scripts/kaggle_save_results.py --branches mf
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
