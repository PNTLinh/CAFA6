# ESM-AlphaFold-GO

Mô hình dự đoán chức năng protein (Gene Ontology) kết hợp **3 nguồn thông tin**:

| Nhánh | Dữ liệu đầu vào | Mạng xử lý |
|---|---|---|
| Cấu trúc protein | Contact map từ AlphaFold2 PDB | GCN + SAGPool (Hierarchical) |
| Trình tự protein | UniProt FASTA → SeqVec/ESM 1024-dim | Average pooling |
| Mạng tương tác PPI | STRING database | GraphSAGE (2 lớp) |

Đầu ra: xác suất cho từng GO term thuộc 3 nhánh **MFO** (308 term), **CCO** (310 term), **BPO** (713 term).

---

## Mục lục

1. [Cấu trúc thư mục](#1-cấu-trúc-thư-mục)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Chuẩn bị dữ liệu thô](#3-chuẩn-bị-dữ-liệu-thô)
4. [Pipeline xử lý dữ liệu](#4-pipeline-xử-lý-dữ-liệu)
5. [Huấn luyện](#5-huấn-luyện)
6. [Đánh giá](#6-đánh-giá)
7. [Kiến trúc model](#7-kiến-trúc-model)
8. [Xử lý sự cố](#8-xử-lý-sự-cố)

---

## 1. Cấu trúc thư mục

```
CAFA6/
├── data_processing/
│   ├── 1_get_valid_ids.py              # Trích xuất protein ID từ file PDB
│   ├── 2_extract_struct_map.py         # (tùy chọn) trích xuất thông tin cấu trúc
│   ├── 3_build_graph_dataset.py        # Ghép graph + seq feature + label → dataset
│   ├── 3_uniprot_mapping.py            # Map ENSP (STRING) → UniProtKB AC
│   ├── 4_build_ppi_graph.py            # Xây PPI global graph từ STRING
│   ├── divide_data.py                  # Chia train/valid/test
│   ├── go_anno.py                      # Parse GO annotation từ GAF file
│   ├── predicted_protein_struct2map.py # PDB → contact map edge list
│   ├── read_seqvec_features.py         # Đọc SeqVec embedding → dict_sequence_feature
│   └── seq2vec.py                      # Tạo SeqVec embedding từ FASTA
│
├── model/
│   ├── network.py    # SAGNetworkHierarchical + PPIEncoder
│   ├── layer.py      # ConvPoolBlock, SAGPool
│   ├── evaluation.py # calculate_performance, cacul_aupr
│   ├── labels_load.py
│   └── utils.py
│
├── train_Struct2GO2.py   # Script huấn luyện chính
├── eval_Struct2GO2.py    # Script đánh giá / xuất kết quả
│
├── proceed_data/         # Dữ liệu đã xử lý (tự động tạo)
├── divided_data/         # Dataset chia train/valid/test (tự động tạo)
├── save_models/          # Model checkpoint (tự động tạo)
├── log/                  # Training log (tự động tạo)
├── test_result/          # Kết quả đánh giá (tự động tạo)
│
├── environment.yml
└── requirements.txt
```

### Dữ liệu thô cần chuẩn bị (xem mục 3)

```
raw_data/
├── struct_feature/           # File PDB từ AlphaFold2 (*.pdb.gz)
│   └── AF-{UniProtID}-F1-model_v4.pdb.gz
├── 9606.protein.links.v12.0.txt  # STRING PPI database
└── goa_human.gaf.gz              # GO annotation (từ UniProt GOA)

# Tải riêng:
go.obo                        # GO ontology definition file
```

---

## 2. Cài đặt môi trường

### Yêu cầu hệ thống

- Python 3.10+
- CUDA GPU (khuyến nghị; có thể chạy CPU nhưng chậm hơn nhiều)
- RAM ≥ 32 GB (PPI graph toàn cục ~20k nodes)

### Tạo môi trường conda

```bash
conda env create -f environment.yml
conda activate cafa6
```

### Hoặc cài thủ công qua pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install biopython scikit-learn transformers tqdm pandas requests
```

### Kiểm tra cài đặt

```python
import torch, dgl
print(torch.__version__)          # >= 2.0
print(dgl.__version__)            # >= 1.1
print(torch.cuda.is_available())  # True nếu có GPU
```

---

## 3. Chuẩn bị dữ liệu thô

### 3.1 Cấu trúc protein — AlphaFold2 PDB

Tải human proteome từ AlphaFold2 database:

```bash
# Tải toàn bộ human proteome (~23k protein, ~50 GB giải nén)
wget https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar

tar -xf UP000005640_9606_HUMAN_v4.tar -C raw_data/struct_feature/
```

Mỗi file có tên dạng `AF-{UniProtID}-F1-model_v4.pdb.gz`.

### 3.2 Trình tự protein — UniProt FASTA

```bash
# Tải FASTA của human proteome từ UniProt
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz
gunzip UP000005640_9606.fasta.gz
mv UP000005640_9606.fasta raw_data/9606-seq.fa
```

### 3.3 GO annotation — UniProt GOA

```bash
wget https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz
mv goa_human.gaf.gz D:/CAFA6/goa_human.gaf.gz
```

### 3.4 GO ontology

```bash
wget https://purl.obolibrary.org/obo/go.obo
```

### 3.5 STRING PPI database

Tải tại: https://string-db.org/cgi/download?species_text=Homo+sapiens

Chọn file: **9606.protein.links.v12.0.txt.gz**

```bash
gunzip 9606.protein.links.v12.0.txt.gz
mv 9606.protein.links.v12.0.txt raw_data/
```

---

## 4. Pipeline xử lý dữ liệu

Chạy theo đúng thứ tự sau. Mỗi bước tạo ra output cho bước tiếp theo.

```
Bước 1 → Bước 2 → Bước 3 → Bước 4 → Bước 5 → Bước 6 → Bước 7 → Bước 8
```

---

### Bước 1 — Trích xuất danh sách protein ID hợp lệ

**Script:** `data_processing/1_get_valid_ids.py`

Quét thư mục `raw_data/struct_feature/`, trích xuất UniProt ID từ tên file PDB.

```bash
python data_processing/1_get_valid_ids.py
```

**Output:** `proceed_data/valid_protein_ids.csv`

---

### Bước 2 — Parse GO annotation

**Script:** `data_processing/go_anno.py`

Đọc file `goa_human.gaf.gz`, lọc annotation theo GO branch (BP/MF/CC), loại bỏ GO term hiếm (tần suất thấp).

```bash
python data_processing/go_anno.py
```

**Output:**
- `proceed_data/human_BP_ACS.json` — `{UniProtID: [GO terms]}`
- `proceed_data/human_MF_ACS.json`
- `proceed_data/human_CC_ACS.json`

> **Ngưỡng lọc GO term:** Sau khi lọc còn 308 term (MFO), 310 term (CCO), 713 term (BPO).

---

### Bước 3 — Xây dựng contact map từ PDB

**Script:** `data_processing/predicted_protein_struct2map.py`

Đọc từng file PDB, tính khoảng cách giữa các C-alpha atom. Hai residue được nối cạnh nếu khoảng cách < 10 Å.

```bash
python data_processing/predicted_protein_struct2map.py
```

**Output:** `proceed_data/proteins_edges/{UniProtID}.txt` — mỗi dòng là một cạnh `src dst`

> Nếu đã có sẵn thư mục `proceed_data/proteins_edges/`, bỏ qua bước này.

---

### Bước 4 — Tạo sequence embedding (SeqVec / ESM)

**Script:** `data_processing/seq2vec.py`

Mã hóa trình tự protein thành vector 1024-dim bằng SeqVec (ELMo-based).

```bash
python data_processing/seq2vec.py \
    -i raw_data/9606-seq.fa \
    -o proceed_data/9606-avg-emb.pkl \
    --protein True
```

Sau đó chuẩn hoá vào dict:

**Script:** `data_processing/read_seqvec_features.py`

Điều chỉnh path trong script rồi chạy:

```bash
python data_processing/read_seqvec_features.py
```

**Output:** `proceed_data/dict_sequence_feature` — `{UniProtID: list(1024,)}`

> **Thay thế bằng ESM-2:** Có thể dùng [ESM-2](https://github.com/facebookresearch/esm) thay SeqVec. Lưu ý output vẫn phải là dict `{UniProtID: ndarray(1024,)}`.

---

### Bước 5 — Map STRING ENSP → UniProtKB (cần cho bước PPI)

**Script:** `data_processing/3_uniprot_mapping.py`

Trích xuất ENSP ID từ STRING file và tra cứu UniProtKB AC qua UniProt REST API.

```bash
# Chỉnh INPUT_FILE trong script trước khi chạy
python data_processing/3_uniprot_mapping.py
```

**Output:** `proceed_data/uniprot_ensembl_mapping.csv` — cột `UniProtKB_AC, Ensembl_Protein`

> Quá trình gửi batch lên UniProt API có thể mất 10–30 phút tuỳ số lượng ENSP ID.

---

### Bước 6 — Xây dựng PPI global graph

**Script:** `data_processing/4_build_ppi_graph.py`

Lọc các cạnh PPI có `combined_score ≥ 700` (high confidence), map ENSP → UniProt, xây dựng DGL graph toàn cục. Node feature là 1024-dim sequence embedding.

```bash
python data_processing/4_build_ppi_graph.py
```

**Output:**
- `proceed_data/ppi_graph_global` — `dgl.DGLGraph` (nodes = protein, edges = PPI)
- `proceed_data/ppi_protein_index` — `{UniProtID: node_id}`

> **Tuỳ chỉnh ngưỡng score:** Mở file và sửa `PPI_SCORE_THRESHOLD = 700`. Giá trị cao hơn → ít cạnh hơn nhưng độ tin cậy cao hơn.

---

### Bước 7 — Ghép tất cả thành graph dataset

**Script:** `data_processing/3_build_graph_dataset.py`

Ghép contact map graph, node feature (node2vec/onehot), sequence feature, GO label và PPI node index cho mỗi protein.

```bash
python data_processing/3_build_graph_dataset.py
```

**Output (cho mỗi nhánh bp / mf / cc):**

| File | Nội dung |
|---|---|
| `proceed_data/emb_graph_{ns}` | `{ID → dgl.DGLGraph}` — contact map graph |
| `proceed_data/emb_seq_feature_{ns}` | `{ID → Tensor(1024,)}` — sequence embedding |
| `proceed_data/emb_label_{ns}` | `{ID → Tensor(num_labels,)}` — multi-hot GO label |
| `proceed_data/emb_ppi_node_id_{ns}` | `{ID → int}` — node index trong PPI graph |
| `proceed_data/label_vocab_{ns}.json` | `[GO term]` — index → GO term |
| `proceed_data/label_{ns}_network` | DGL graph co-occurrence của GO label |

> **Node feature mode:** Mở script và đặt `NODE_FEAT_MODE`:
> - `"node2vec"` (30-dim) → dùng với `train_Struct2GO.py` (input_dim=30)
> - `"onehot"` (26-dim) → one-hot amino acid per residue
> - `"concat"` (56-dim) → ghép cả hai, dùng với `train_Struct2GO2.py` (input_dim=56)

---

### Bước 8 — Chia train / valid / test

**Script:** `data_processing/divide_data.py`

Chia ngẫu nhiên theo tỷ lệ 70% / 20% / 10%.

```bash
python data_processing/divide_data.py
```

**Output:**
- `divided_data/{ns}_train_dataset`
- `divided_data/{ns}_valid_dataset`
- `divided_data/{ns}_test_dataset`

---

## 5. Huấn luyện

Tạo trước các thư mục cần thiết:

```bash
mkdir -p save_models log test_result
```

### Chạy huấn luyện

```bash
# Nhánh MF (Molecular Function) — 308 label
python train_Struct2GO2.py -branch mf -labels_num 308 -batch_size 32 -learningrate 1e-4 -dropout 0.2

# Nhánh CC (Cellular Component) — 310 label
python train_Struct2GO2.py -branch cc -labels_num 310 -batch_size 32 -learningrate 1e-4 -dropout 0.2

# Nhánh BP (Biological Process) — 713 label
python train_Struct2GO2.py -branch bp -labels_num 713 -batch_size 32 -learningrate 1e-4 -dropout 0.1
```

### Tham số huấn luyện

| Tham số | Mô tả | Mặc định |
|---|---|---|
| `-branch` | Nhánh GO: `bp`, `mf`, `cc` | `mf` |
| `-labels_num` | Số GO term của nhánh đó | `328` |
| `-batch_size` | Batch size | `64` |
| `-learningrate` | Learning rate (Adam) | `1e-4` |
| `-dropout` | Dropout rate | `0.3` |

### Theo dõi training

```bash
# Xem log real-time
tail -f log/mf.log
```

**Checkpoint tốt nhất** được lưu tự động tại:

```
save_models/bestmodel_{branch}_{batch}_{lr}_{dropout}.pkl
```

> **Điều chỉnh GPU:** Mở `train_Struct2GO2.py` và sửa `device = "cuda:0"` thành GPU phù hợp.

---

## 6. Đánh giá

```bash
# Đánh giá nhánh MF với threshold = 0.71
python eval_Struct2GO2.py -branch mf -thresh 0.71

# Đánh giá nhánh CC
python eval_Struct2GO2.py -branch cc -thresh 0.50

# Đánh giá nhánh BP
python eval_Struct2GO2.py -branch bp -thresh 0.40
```

### Output đánh giá

| File | Nội dung |
|---|---|
| `test_result/{branch}_result.json` | GO term mới dự đoán cho từng protein |
| `test_result/{branch}_roc_curve.png` | Biểu đồ ROC curve |
| `log/test_{branch}.log` | Metrics: Loss, F-max, AUC, AUPR, Precision, Recall |

### Ý nghĩa metrics

| Metric | Mô tả |
|---|---|
| **F-max** | F1-score tối đa trên tập ngưỡng 0.01–0.99 |
| **AUC** | Area Under ROC Curve |
| **AUPR** | Area Under Precision-Recall Curve |

---

## 7. Kiến trúc model

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SAGNetworkHierarchical                          │
│                                                                     │
│  Protein Structure                                                  │
│  (contact graph)   ──► ConvPoolBlock ──► ConvPoolBlock ──► [hid×2] ─┐
│                         (GCN+SAGPool)   (GCN+SAGPool)               │
│                                                                      │
│  Protein Sequence                                                    │
│  (ESM/SeqVec 1024) ──────────────────────────────────► [1024]     ──┤ Concat
│                                                                      │ [hid×2+1024+256]
│  PPI Network                                                         │
│  (STRING graph)    ──► GraphSAGE ──► GraphSAGE ──────► [256]      ──┘
│                         (PPIEncoder)                    │
│                                                         ▼
│                                                   Linear(hid×2)
│                                                   Linear(hid)
│                                                   Linear(num_labels)
│                                                         │
│                                                         ▼
│                                               GO term predictions
└─────────────────────────────────────────────────────────────────────┘
```

### Tham số model mặc định

| Thành phần | Tham số |
|---|---|
| `in_dim` | 56 (onehot 26 + node2vec 30) |
| `hid_dim` | 512 |
| `num_convs` | 6 |
| `pool_ratio` | 0.75 |
| `ppi_in_dim` | 1024 (seq embedding) |
| `ppi_hid_dim` | 512 |
| `ppi_out_dim` | 256 |

---

## 8. Xử lý sự cố

### Lỗi thường gặp

**`FileNotFoundError: ppi_graph_global`**
```
Chưa chạy bước 6. Chạy: python data_processing/4_build_ppi_graph.py
```

**`FileNotFoundError: uniprot_ensembl_mapping.csv`**
```
Chưa chạy bước 5. Chạy: python data_processing/3_uniprot_mapping.py
```

**`KeyError` hoặc dimension mismatch trong model**
```
Đảm bảo NODE_FEAT_MODE trong 3_build_graph_dataset.py khớp với in_dim của model:
- "node2vec" (30-dim) → train_Struct2GO.py
- "concat"   (56-dim) → train_Struct2GO2.py
```

**`CUDA out of memory`**
```
Giảm batch_size: -batch_size 16
Hoặc giảm pool_ratio trong model: pool_ratio=0.5
```

**UniProt API timeout (bước 5)**
```
Script tự retry. Nếu vẫn lỗi, giảm BATCH_SIZE trong 3_uniprot_mapping.py xuống 200.
```

**Protein bị bỏ qua (skipped) nhiều ở bước 7**
```
Kiểm tra xem bước 3 đã chạy đủ protein chưa:
ls proceed_data/proteins_edges/ | wc -l
```

### Hard-coded path cần sửa

Một số script còn path cũ cần cập nhật trước khi chạy:

| Script | Path cần sửa |
|---|---|
| `data_processing/predicted_protein_struct2map.py` | `/mnt/workspace/...` → path thực |
| `data_processing/read_seqvec_features.py` | `/mnt/workspace/...` → path thực |
| `train_Struct2GO2.py` | `/etc/dsw/divided_data/...` → `divided_data/` |
| `eval_Struct2GO2.py` | `/etc/dsw/divided_data/...` → `divided_data/` |

### Tóm tắt thứ tự chạy đầy đủ

```bash
# 1. Lấy protein ID hợp lệ
python data_processing/1_get_valid_ids.py

# 2. Parse GO annotation
python data_processing/go_anno.py

# 3. Xây contact map từ PDB
python data_processing/predicted_protein_struct2map.py

# 4. Tạo sequence embedding
python data_processing/seq2vec.py -i raw_data/9606-seq.fa -o proceed_data/9606-avg-emb.pkl --protein True
python data_processing/read_seqvec_features.py

# 5. Map STRING ENSP → UniProt
python data_processing/3_uniprot_mapping.py

# 6. Xây PPI graph
python data_processing/4_build_ppi_graph.py

# 7. Ghép thành dataset
python data_processing/3_build_graph_dataset.py

# 8. Chia train/valid/test
python data_processing/divide_data.py

# 9. Huấn luyện
python train_Struct2GO2.py -branch mf -labels_num 308 -batch_size 32 -learningrate 1e-4 -dropout 0.2
python train_Struct2GO2.py -branch cc -labels_num 310 -batch_size 32 -learningrate 1e-4 -dropout 0.2
python train_Struct2GO2.py -branch bp -labels_num 713 -batch_size 32 -learningrate 1e-4 -dropout 0.1

# 10. Đánh giá
python eval_Struct2GO2.py -branch mf -thresh 0.71
python eval_Struct2GO2.py -branch cc -thresh 0.50
python eval_Struct2GO2.py -branch bp -thresh 0.40
```
