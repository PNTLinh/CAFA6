import argparse
import logging
import os
import pickle
import warnings
from typing import Any

# Patch DGL on disk before import (Kaggle Py3.12 / torchdata break)
from model.dgl_patch import ensure_dgl_importable

ensure_dgl_importable(verbose=False)

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr, calculate_performance
from model.network import SAGNetworkHierarchical

warnings.filterwarnings("ignore")


def _register_pickle_classes() -> None:
    """Pickle from `python divide_data.py` references __main__.MyDataSet."""
    import __main__

    __main__.MyDataSet = MyDataSet


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "MyDataSet" and module in {"__main__", "data_processing.divide_data"}:
            return MyDataSet
        return super().find_class(module, name)


def _load_pickle(path: str):
    with open(path, "rb") as handle:
        return _CompatUnpickler(handle).load()
Thresholds = [x / 100 for x in range(1, 100)]


def create_logger(branch_name: str) -> logging.Logger:
    os.makedirs("log", exist_ok=True)
    os.makedirs("save_models", exist_ok=True)
    logger = logging.getLogger(branch_name)
    if logger.handlers:
        return logger
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join("log", branch_name + ".log"))
    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.ERROR)
    handler2.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def resolve_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    # Kaggle/Linux: CUDA sẵn có. Windows: cần DGL_CUDA=1 nếu đã cài DGL CUDA wheel.
    on_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    dgl_cuda = os.environ.get("DGL_CUDA", "1" if on_kaggle else "0") == "1"
    if dgl_cuda or on_kaggle:
        return torch.device("cuda:0")
    return torch.device("cpu")


def apply_kaggle_preset(args: argparse.Namespace) -> None:
    """Preset tối ưu cho Kaggle GPU T4 (16GB)."""
    args.batch_size = 96
    args.epochs = 20
    args.num_workers = 2
    args.hid_dim = 384
    args.num_convs = 4
    args.ppi_out_dim = 128
    args.amp = True
    args.cache_ppi = True
    args.validate_every = 2


def labels_to_device(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    labels = torch.squeeze(labels)
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(0)
    return labels.to(device).float()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=64)
    parser.add_argument("-learningrate", "--learningrate", type=float, default=1e-4)
    parser.add_argument("-dropout", "--dropout", type=float, default=0.3)
    parser.add_argument("-branch", "--branch", type=str, default="mf", choices=["bp", "mf", "cc"])
    parser.add_argument("-labels_num", "--labels_num", type=int, default=328)
    parser.add_argument("-epochs", "--epochs", type=int, default=10)
    parser.add_argument("-hid_dim", "--hid_dim", type=int, default=256)
    parser.add_argument("-num_convs", "--num_convs", type=int, default=3)
    parser.add_argument("-seq_dim", "--seq_dim", type=int, default=640)
    parser.add_argument("-ppi_out_dim", "--ppi_out_dim", type=int, default=128)
    parser.add_argument("-num_workers", "--num_workers", type=int, default=4)
    parser.add_argument("-validate_every", "--validate_every", type=int, default=4,
                        help="Validate mỗi N epoch (1 = mỗi epoch)")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (FP16) — khuyến nghị trên T4")
    parser.add_argument("--no_cache_ppi", dest="cache_ppi", action="store_false",
                        help="Tắt cache PPI embedding mỗi epoch")
    parser.set_defaults(cache_ppi=True)
    parser.add_argument("--cpu", action="store_true", help="Bắt buộc train trên CPU")
    parser.add_argument("--kaggle", action="store_true",
                        help="Preset T4: batch=96, hid=384, conv=4, amp, cache PPI")
    args = parser.parse_args()

    if args.kaggle:
        apply_kaggle_preset(args)

    device = resolve_device(force_cpu=args.cpu)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    data_dir = os.environ.get("DATA_DIR", "D:/CAFA6")
    train_data_path = f"{data_dir}/divided_data/{args.branch}_train_dataset"
    valid_data_path = f"{data_dir}/divided_data/{args.branch}_valid_dataset"
    label_network_path = f"{data_dir}/proceed_data/label_{args.branch}_network"
    ppi_graph_path = f"{data_dir}/proceed_data/ppi_graph_global"

    logger = create_logger(args.branch)
    logger.info(f"device={device}, amp={args.amp}, cache_ppi={args.cache_ppi}, kaggle={args.kaggle}")

    _register_pickle_classes()
    train_dataset = _load_pickle(train_data_path)
    valid_dataset = _load_pickle(valid_data_path)
    label_network = _load_pickle(label_network_path)
    label_network = label_network.to(device)

    ppi_graph = _load_pickle(ppi_graph_path)
    ppi_graph = ppi_graph.to(device)

    sample_label = train_dataset[0][2]
    detected_labels = int(np.asarray(sample_label).reshape(-1).shape[0])
    if detected_labels != args.labels_num:
        print(f"[INFO] Auto-detected labels_num = {detected_labels} (override CLI {args.labels_num})")
        labels_num = detected_labels
    else:
        labels_num = args.labels_num

    sample_seq = np.asarray(train_dataset[0][3]).reshape(-1)
    args.seq_dim = int(sample_seq.shape[0])
    ppi_feat_dim = int(ppi_graph.ndata["feat"].shape[1])
    if ppi_feat_dim != args.seq_dim:
        print(f"[INFO] ppi_in_dim={ppi_feat_dim} (from ppi_graph.ndata['feat'])")

    loader_kw = dict(
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True

    train_dataloader = GraphDataLoader(dataset=train_dataset, shuffle=True, **loader_kw)
    valid_dataloader = GraphDataLoader(dataset=valid_dataset, shuffle=False, **loader_kw)

    model = SAGNetworkHierarchical(
        56,
        args.hid_dim,
        labels_num,
        num_convs=args.num_convs,
        pool_ratio=0.5,
        dropout=args.dropout,
        seq_dim=args.seq_dim,
        ppi_in_dim=ppi_feat_dim,
        ppi_hid_dim=args.hid_dim,
        ppi_out_dim=args.ppi_out_dim,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learningrate, weight_decay=1e-4)
    total_steps = args.epochs * max(len(train_dataloader), 1)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(200, total_steps // 10), num_training_steps=total_steps
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda and args.amp)

    best_fscore = 0.0
    best_aupr = 0.0
    best_scores = []
    best_score_dict = {}

    logger.info("#########" + args.branch + "###########")
    logger.info("########start training###########")

    for epoch in range(args.epochs):
        print("epoch:", epoch)
        logger.info("epoch: " + str(epoch))
        model.train()

        ppi_node_emb = None
        if args.cache_ppi:
            with torch.cuda.amp.autocast(enabled=use_cuda and args.amp):
                ppi_node_emb = model.encode_ppi_nodes(ppi_graph)

        train_loss = 0.0
        for i, (_, graphs, labels, seq_feats, ppi_node_ids) in enumerate(
            tqdm(train_dataloader, desc=f"train e{epoch}")
        ):
            graphs = graphs.to(device, non_blocking=use_cuda)
            seq_feats = seq_feats.to(device, non_blocking=use_cuda)
            labels = labels_to_device(labels, device)
            ppi_node_ids = ppi_node_ids.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_cuda and args.amp):
                logits = model(
                    graphs,
                    seq_feats,
                    label_network,
                    ppi_graph=ppi_graph,
                    ppi_node_ids=ppi_node_ids,
                    ppi_node_emb=ppi_node_emb,
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            train_loss += loss.item()
            if i % 30 == 29:
                logger.info(
                    f"Epoch {epoch}/{args.epochs}, step {i}/{len(train_dataloader)}, loss={loss.item():.4f}"
                )

        run_valid = (epoch + 1) % args.validate_every == 0 or epoch == args.epochs - 1
        if not run_valid:
            continue

        model.eval()
        print("validating")
        logger.info("validating")
        valid_loss = 0.0
        pred, actual = [], []

        if args.cache_ppi:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda and args.amp):
                ppi_node_emb = model.encode_ppi_nodes(ppi_graph)

        with torch.no_grad():
            for i, (_, graphs, labels, seq_feats, ppi_node_ids) in enumerate(
                tqdm(valid_dataloader, desc=f"valid e{epoch}")
            ):
                graphs = graphs.to(device, non_blocking=use_cuda)
                seq_feats = seq_feats.to(device, non_blocking=use_cuda)
                labels = labels_to_device(labels, device)
                ppi_node_ids = ppi_node_ids.to(device, non_blocking=use_cuda)

                with torch.cuda.amp.autocast(enabled=use_cuda and args.amp):
                    logits = model(
                        graphs,
                        seq_feats,
                        label_network,
                        ppi_graph=ppi_graph,
                        ppi_node_ids=ppi_node_ids,
                        ppi_node_emb=ppi_node_emb,
                    )
                    loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)

                valid_loss += loss.item()
                pred += probs.tolist()
                actual += labels.tolist()

        fpr, tpr, _ = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
        auc_score = auc(fpr, tpr)
        aupr = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())

        each_best_fcore = 0.0
        each_best_scores = []
        score_dict = {}
        for thresh in Thresholds:
            f_score, precision, recall = calculate_performance(
                actual, pred, label_network, threshold=thresh
            )
            if f_score >= each_best_fcore:
                each_best_fcore = f_score
                each_best_scores = [thresh, f_score, recall, precision, auc_score]
                score_dict[thresh] = [f_score, recall, precision, auc_score]

        if each_best_fcore >= best_fscore:
            best_fscore = each_best_fcore
            best_scores = each_best_scores
            best_score_dict = score_dict
            best_aupr = aupr
            ckpt = (
                f"save_models/bestmodel_{args.branch}_{args.batch_size}_"
                f"{args.learningrate}_{args.dropout}.pkl"
            )
            torch.save(model, ckpt)
            logger.info(f"saved checkpoint: {ckpt}")

        thresh, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
        precision, auc_score = each_best_scores[3], each_best_scores[4]
        logger.info("########valid metric###########")
        logger.info(
            f"epoch={epoch}, train_loss={train_loss / len(train_dataloader):.4f}, "
            f"valid_loss={valid_loss / len(valid_dataloader):.4f}"
        )
        logger.info(
            f"threshold={thresh}, f_score={f_score}, auc={auc_score}, "
            f"recall={recall}, precision={precision}, aupr={aupr}"
        )

    logger.info("best_fscore: " + str(best_fscore))
    logger.info("best_scores[thresh,fmax,recall,precision,auc]: " + str(best_scores))
    logger.info("best_aupr: " + str(best_aupr))
    logger.info("best_score_dict: " + str(best_score_dict))


if __name__ == "__main__":
    main()
