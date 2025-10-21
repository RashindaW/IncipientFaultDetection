"""
Quick-training script for DyEdgeGAT using a limited subset of CSVs.

This is useful for fast sanity checks without loading the full dataset.

Example:
    conda run -n rashindaNew-torch-env python fast_train_dyedgegat.py \
        --epochs 5 --train-files BaselineTestA.csv BaselineTestC.csv \
        --val-file BaselineTestB.csv --test-file Fault1_DisplayCaseDoorOpen.csv \
        --device auto
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dyedgegat"))

from src.config import cfg
from src.data.dataset import RefrigerationDataset
from src.data.column_config import MEASUREMENT_VARS, CONTROL_VARS
from src.model.dyedgegat import DyEdgeGAT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast DyEdgeGAT training on subset of data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for DataLoaders")
    parser.add_argument("--train-stride", type=int, default=10, help="Stride for training windows")
    parser.add_argument("--val-stride", type=int, default=20, help="Stride for validation windows")
    parser.add_argument("--test-stride", type=int, default=20, help="Stride for test windows")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--data-dir", type=str, default="Dataset", help="Directory containing CSV files")
    parser.add_argument(
        "--train-files",
        nargs="+",
        default=["BaselineTestA.csv", "BaselineTestC.csv", "BaselineTestD.csv"],
        help="One or more CSV files for training",
    )
    parser.add_argument("--val-file", type=str, default="BaselineTestB.csv", help="CSV file for validation")
    parser.add_argument("--test-file", type=str, default="Fault1_DisplayCaseDoorOpen.csv", help="CSV file for testing")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device ('auto' selects CUDA if available).",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index (requires CUDA to be available).",
    )
    return parser.parse_args()


def resolve_device(device_flag: str, cuda_index: Optional[int]) -> torch.device:
    if device_flag == "cpu":
        if cuda_index is not None:
            raise ValueError("CUDA device index specified but device='cpu'.")
        return torch.device("cpu")

    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if cuda_index is not None:
            if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
                raise ValueError(f"Requested CUDA device {cuda_index}, but only {torch.cuda.device_count()} devices are visible.")
            torch.cuda.set_device(cuda_index)
            return torch.device(f"cuda:{cuda_index}")
        return torch.device("cuda")

    # auto
    if torch.cuda.is_available():
        if cuda_index is not None:
            if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
                raise ValueError(f"Requested CUDA device {cuda_index}, but only {torch.cuda.device_count()} devices are visible.")
            torch.cuda.set_device(cuda_index)
            return torch.device(f"cuda:{cuda_index}")
        return torch.device("cuda")

    if cuda_index is not None:
        raise ValueError("CUDA device index specified but CUDA is not available.")
    return torch.device("cpu")


def init_model(device: torch.device) -> DyEdgeGAT:
    cfg.set_dataset_params(
        n_nodes=len(MEASUREMENT_VARS),
        window_size=15,
        ocvar_dim=len(CONTROL_VARS),
    )
    cfg.device = str(device)
    cfg.validate()

    model = DyEdgeGAT(
        feat_input_node=1,
        feat_target_node=1,
        feat_input_edge=1,
        node_encoder_type="gru",
        node_encoder_mode="univariate",
        contr_encoder_type="gru",
        infer_temporal_edge=True,
        temp_edge_hid_dim=100,
        temp_edge_embed_dim=1,
        temporal_window=5,
        temporal_kernel=5,
        use_time_encoding=True,
        time_dim=5,
        temp_node_embed_dim=16,
        infer_static_graph=True,
        feat_edge_hid_dim=128,
        topk=20,
        learn_sys=True,
        num_gnn_layers=2,
        gnn_embed_dim=40,
        gnn_type="gin",
        dropout=0.3,
        do_encoder_norm=True,
        do_gnn_norm=True,
        do_decoder_norm=True,
        encoder_norm_type="layer",
        gnn_norm_type="layer",
        decoder_norm_type="layer",
        recon_hidden_dim=16,
        num_recon_layers=1,
        edge_aggr="temp",
        act="relu",
        aug_control=True,
        flip_output=True,
    )
    return model.to(device)


def build_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = RefrigerationDataset(
        data_files=args.train_files,
        window_size=cfg.dataset.window_size,
        stride=args.train_stride,
        data_dir=args.data_dir,
        normalize=True,
    )
    norm_stats = train_dataset.get_normalization_stats()

    val_dataset = RefrigerationDataset(
        data_files=[args.val_file],
        window_size=cfg.dataset.window_size,
        stride=args.val_stride,
        data_dir=args.data_dir,
        normalize=True,
        normalization_stats=norm_stats,
    )

    test_dataset = RefrigerationDataset(
        data_files=[args.test_file],
        window_size=cfg.dataset.window_size,
        stride=args.test_stride,
        data_dir=args.data_dir,
        normalize=True,
        normalization_stats=norm_stats,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def compute_recon_loss(model: DyEdgeGAT, batch, criterion, device) -> torch.Tensor:
    batch = batch.to(device)
    recon = model(batch)
    target = batch.x.unsqueeze(-1)
    return criterion(recon, target)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = compute_recon_loss(model, batch, criterion, device)
        loss.backward()
        optimizer.step()

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_score = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        recon, edge_index, edge_attr = model(batch, return_graph=True)
        target = batch.x.unsqueeze(-1)

        loss = criterion(recon, target)
        score = model.compute_topology_aware_anomaly_score(
            target, recon, edge_index, edge_attr
        )

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_score += score.item() * batch_size
        total_samples += batch_size

    denom = max(total_samples, 1)
    return total_loss / denom, total_score / denom


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, args.cuda_device)

    print("=" * 80)
    print("FAST DYEDGEGAT TRAINING")
    print("=" * 80)
    print(f"Train files: {args.train_files}")
    print(f"Val file  : {args.val_file}")
    print(f"Test file : {args.test_file}")
    print(f"Device    : {device}")
    print("=" * 80)

    model = init_model(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader, val_loader, test_loader = build_dataloaders(args)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_anom={val_score:.6f} time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest validation loss: {best_val_loss:.6f}")
    else:
        print("\nWarning: No improvement over initial epoch.")

    test_loss, test_score = evaluate(model, test_loader, criterion, device)
    print(
        f"\nTest results on {args.test_file}: "
        f"recon_loss={test_loss:.6f} anomaly_score={test_score:.6f}"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
