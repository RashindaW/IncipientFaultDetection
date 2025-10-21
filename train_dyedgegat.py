"""
Training script for DyEdgeGAT on the refrigeration system dataset.

Usage:
    conda run -n rashindaNew-torch-env python train_dyedgegat.py --epochs 10
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dyedgegat"))

from src.config import cfg
from src.data.dataloader import create_dataloaders
from src.data.column_config import MEASUREMENT_VARS, CONTROL_VARS
from src.model.dyedgegat import DyEdgeGAT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DyEdgeGAT on refrigeration dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--train-stride", type=int, default=1, help="Sliding window stride for training dataset")
    parser.add_argument("--val-stride", type=int, default=5, help="Sliding window stride for validation/test datasets")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument("--data-dir", type=str, default="Dataset", help="Directory containing CSV files")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device. 'auto' selects CUDA if available.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index (e.g., 0, 1, 2). Requires --device cuda/auto with CUDA available.",
    )
    parser.add_argument("--save-model", type=str, default=None, help="Optional path to save best model state_dict")
    return parser.parse_args()


def resolve_device(device_flag: str, cuda_index: Optional[int]) -> torch.device:
    if device_flag == "cpu":
        if cuda_index is not None:
            raise ValueError("A CUDA device index was provided but device='cpu'.")
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


def compute_recon_loss(model: DyEdgeGAT, batch: torch.Tensor, criterion: torch.nn.Module, device: torch.device) -> torch.Tensor:
    batch = batch.to(device)
    recon = model(batch)
    target = batch.x.unsqueeze(-1)
    loss = criterion(recon, target)
    return loss


def train_epoch(
    model: DyEdgeGAT,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = compute_recon_loss(model, batch, criterion, device)
        loss.backward()
        optimizer.step()

        batch_size = batch.num_graphs
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate(
    model: DyEdgeGAT,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_score = 0.0
    sample_count = 0

    for batch in loader:
        batch = batch.to(device)
        recon, edge_index, edge_attr = model(batch, return_graph=True)
        target = batch.x.unsqueeze(-1)

        loss = criterion(recon, target)
        score = model.compute_topology_aware_anomaly_score(
            target, recon, edge_index, edge_attr
        )

        batch_size = batch.num_graphs
        running_loss += loss.item() * batch_size
        running_score += score.item() * batch_size
        sample_count += batch_size

    denom = max(sample_count, 1)
    return running_loss / denom, running_score / denom


@torch.no_grad()
def evaluate_tests(
    model: DyEdgeGAT,
    loaders: Dict[str, torch.utils.data.DataLoader],
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name, loader in loaders.items():
        loss, score = evaluate(model, loader, criterion, device)
        metrics[name] = {"recon_loss": loss, "anomaly_score": score}
    return metrics


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, args.cuda_device)
    print("=" * 80)
    print(f"Training DyEdgeGAT on device: {device}")
    print("=" * 80)

    model = init_model(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader, val_loader, test_loaders = create_dataloaders(
        window_size=cfg.dataset.window_size,
        batch_size=args.batch_size,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        data_dir=args.data_dir,
        num_workers=0,
    )

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        history.append((epoch, train_loss, val_loss))
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

    print("\nEvaluating on test datasets...")
    test_scores = evaluate_tests(model, test_loaders, criterion, device)
    for name, metrics in test_scores.items():
        print(
            f"  {name:30s}: recon_loss={metrics['recon_loss']:.6f} "
            f"anomaly_score={metrics['anomaly_score']:.6f}"
        )

    if args.save_model:
        torch.save(best_state if best_state is not None else model.state_dict(), args.save_model)
        print(f"\nSaved model checkpoint to: {args.save_model}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
