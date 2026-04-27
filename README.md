# ESM-AlphaFold-GO

This repository contains the data processing, training, and evaluation pipeline for protein function prediction with graph + sequence features.

## 1) Environment

- Main environment file: `environment.yml`
- Optional dependency list: `requirements.txt`
- Suggested platform: CUDA GPU (the training scripts currently use fixed CUDA device ids)

Create environment (example with conda):

```bash
conda env create -f environment.yml
conda activate <your_env_name>
```

## 2) Data Processing Pipeline

This section starts from raw data and builds train/valid/test datasets.

### 2.1 Prepare required inputs

Expected resources:

- STRING links file (example): `9606.protein.links.v12.0.txt`
- Protein structure/contact inputs (PDB or precomputed edge files)
- Sequence features (SeqVec/ESM-derived vectors)
- GO annotations for three branches:
    - `proceed_data/human_BP_ACS.json`
    - `proceed_data/human_MF_ACS.json`
    - `proceed_data/human_CC_ACS.json`

### 2.2 UniProt mapping from STRING ENSP ids

Script: `data_processing/uniprot_mapping.py`

Purpose:

- Extract unique `ENSP...` IDs from STRING file
- Submit ID mapping jobs to UniProt REST API
- Save CSV mapping: `UniProtKB_AC, Ensembl_Protein`

Run:

```bash
python data_processing/uniprot_mapping.py
```

Note: adjust `INPUT_FILE` in the script to your local STRING txt path.

### 2.3 Build protein contact map edges (if not already available)

Script: `data_processing/predicted_protein_struct2map.py`

Purpose:

- Read PDB files
- Build residue-residue contacts (distance threshold)
- Export edge list txt files

If you already have `proceed_data/proteins_edges/*.txt`, you can skip this step.

### 2.4 Build node and sequence features

Related files:

- `data_processing/node2vec.ipynb`
- `data_processing/read_node2vec_feature.py`
- `data_processing/ESeq2vec.ipynb`
- `data_processing/read_seqvec_features.py`

Expected outputs used by next step:

- `proceed_data/protein_node2vec`
- `proceed_data/protein_node2onehot`
- `proceed_data/dict_sequence_feature`

### 2.5 Build graph dataset per GO branch

Script: `data_processing/3_build_graph_dataset.py`

Purpose:

- Merge contact-map graph, node features, sequence features, and GO labels
- Build branch-specific artifacts for `bp`, `mf`, `cc`

Run:

```bash
python data_processing/3_build_graph_dataset.py
```

Main outputs (per branch):

- `proceed_data/emb_graph_{ns}`
- `proceed_data/emb_seq_feature_{ns}`
- `proceed_data/emb_label_{ns}`
- `proceed_data/label_vocab_{ns}.json`
- `proceed_data/label_{ns}_network`

### 2.6 Split into train/valid/test

Script: `data_processing/divide_data.py`

Run:

```bash
python data_processing/divide_data.py
```

Outputs:

- `divided_data/bp_train_dataset`, `divided_data/bp_valid_dataset`, `divided_data/bp_test_dataset`
- `divided_data/mf_train_dataset`, `divided_data/mf_valid_dataset`, `divided_data/mf_test_dataset`
- `divided_data/cc_train_dataset`, `divided_data/cc_valid_dataset`, `divided_data/cc_test_dataset`

## 3) Training

Main scripts:

- `train_Struct2GO.py`
- `train_Struct2GO2.py`

Important arguments:

- `-branch`: one of `bp`, `mf`, `cc`
- `-labels_num`: number of labels for that branch
- `-batch_size`, `-learningrate`, `-dropout`

Examples:

```bash
python train_Struct2GO2.py -branch mf -labels_num 308 -batch_size 32 -learningrate 1e-4 -dropout 0.2
python train_Struct2GO2.py -branch cc -labels_num 310 -batch_size 32 -learningrate 1e-4 -dropout 0.2
python train_Struct2GO2.py -branch bp -labels_num 713 -batch_size 32 -learningrate 1e-4 -dropout 0.1
```

Training outputs:

- Best model: `save_models/bestmodel_{branch}_{batch}_{lr}_{dropout}.pkl`
- Logs: `log/{branch}.log`

You can also check command history in `run_train.sh`.

## 4) Evaluation and Prediction Export

Main scripts:

- `eval_Struct2GO.py`
- `eval_Struct2GO2.py`

Example:

```bash
python eval_Struct2GO2.py -branch mf -thresh 0.71
```

Outputs:

- Predicted new GO labels: `test_result/{branch}_result.json`
- Metrics printed/logged: loss, F-score, AUC, AUPR
- ROC image (Struct2GO2 eval): `test_result/{branch}_roc_curve.png`

## 5) Notes and Troubleshooting

- Some scripts still contain hard-coded paths from older environments (Linux paths like `/mnt/...` or `/etc/dsw/...`). Update paths before running in a new machine.
- Some scripts fix GPU id in code (for example `cuda:0` / `cuda:7`). Change it to match your environment.
- Keep feature dimensions consistent with model input:
    - `train_Struct2GO.py` currently uses input dim `30`
    - `train_Struct2GO2.py` currently uses input dim `56`
    Ensure `NODE_FEAT_MODE` in `data_processing/3_build_graph_dataset.py` matches your selected training script.
