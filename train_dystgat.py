
import argparse
import os
import random
import time
import csv
import math
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    accuracy_score,
)

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as TorchDDP
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dystgat"))

from src.config import cfg
from datasets import get_adapter, list_adapter_keys
from src.model.dystgat import DySTGAT
from src.utils.checkpoint import EpochCheckpointManager
from src.utils.tea import TemporalEvidenceAccumulator, compute_tea_metrics


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full determinism (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    available_datasets = list_adapter_keys()
    parser = argparse.ArgumentParser(description="Train DySTGAT (Dynamic Spectral-Temporal GAT) for anomaly detection")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not set, runs are non-deterministic.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--train-stride", type=int, default=1, help="Sliding window stride for training dataset")
    parser.add_argument("--val-stride", type=int, default=5, help="Sliding window stride for validation/test datasets")
    parser.add_argument(
        "--test-stride",
        type=int,
        default=None,
        help="Optional sliding window stride for test datasets (defaults to --val-stride).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Temporal window size (number of timesteps) for sliding windows.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["reconstruction", "prediction"],
        default="reconstruction",
        help="Training task: reconstruct the input window or predict future horizon.",
    )
    parser.add_argument(
        "--pred-horizon",
        type=int,
        default=0,
        help="Prediction horizon (required when --task prediction).",
    )
    parser.add_argument(
        "--anomaly-weight",
        type=float,
        default=0.0,
        help="Weight for topology-aware anomaly score penalty added to training loss (0 disables).",
    )
    parser.add_argument(
        "--use-spectral-view",
        action="store_true",
        help="Enable dual-view spectral branch with spectral graph and divergence loss.",
    )
    parser.add_argument(
        "--freq-embed-dim",
        type=int,
        default=16,
        help="Embedding dimension for spectral encoder/GNN path.",
    )
    parser.add_argument(
        "--freq-bins",
        type=int,
        default=0,
        help="Number of rFFT bins to keep (0 uses all window_size//2+1).",
    )
    parser.add_argument(
        "--freq-band-mix",
        type=str,
        default="none",
        choices=["none", "conv", "mlp"],
        help="Optional band-mixing layer on spectral bins.",
    )
    parser.add_argument(
        "--freq-use-log",
        dest="freq_use_log",
        action="store_true",
        help="Use log-magnitude scaling for spectral inputs (default).",
    )
    parser.add_argument(
        "--no-freq-use-log",
        dest="freq_use_log",
        action="store_false",
        help="Disable log-magnitude scaling for spectral inputs.",
    )
    parser.set_defaults(freq_use_log=True)
    parser.add_argument(
        "--freq-use-spectral-features",
        action="store_true",
        help="Append spectral shape features (centroid/flatness/rolloff/bands) to spectral embeddings.",
    )
    parser.add_argument(
        "--freq-topk",
        type=int,
        default=None,
        help="Optional top-k neighbors for spectral graph (defaults to temporal topk).",
    )
    parser.add_argument(
        "--share-gnn-weights",
        action="store_true",
        help="Reuse temporal GNN weights for spectral branch (otherwise separate GNN stack).",
    )
    parser.add_argument(
        "--fuse-mode",
        type=str,
        default="concat",
        choices=["concat", "sum", "gated"],
        help="Fusion strategy for temporal and spectral node embeddings.",
    )
    parser.add_argument(
        "--divergence-type",
        type=str,
        default="js",
        choices=["js", "kl"],
        help="Divergence metric between temporal/spectral attentions.",
    )
    parser.add_argument(
        "--lambda-div",
        type=float,
        default=0.0,
        help="Weight for spectral/temporal divergence loss during training.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay (L2 regularization)")
    parser.add_argument(
        "--dataset-key",
        type=str,
        choices=available_datasets,
        help=f"Dataset adapter to use. Available: {', '.join(available_datasets)}.",
    )
    parser.add_argument(
        "--ashrae-feature-option",
        type=str,
        choices=["a", "b"],
        default="a",
        help="Feature selection for ASHRAE only: 'a' (minimal context) or 'b' (control-aware).",
    )
    parser.add_argument(
        "--ashrae-faults",
        type=str,
        default="all",
        help="ASHRAE only: comma-separated fault keys to include in testing (default 'all'). "
             "Valid keys: Condenser_Fouling_06, Condenser_Fouling_12, Condenser_Fouling_20, "
             "Condenser_Fouling_30, Condenser_Fouling_45.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override dataset directory (defaults to adapter recommendation).",
    )
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
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="Comma separated CUDA device indices (single value only, e.g., '0').",
    )
    parser.add_argument("--save-model", type=str, default=None, help="Optional path to save best model state_dict")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to store per-epoch checkpoints and metrics CSV.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation using a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to load before evaluation or to warm-start training.",
    )
    parser.add_argument(
        "--baseline-from",
        type=str,
        choices=["val", "train", "test"],
        default="test",
        help="Source split for the baseline (normal) test loader; default uses benchmark test split.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip evaluation on test/fault datasets (train + validation only).",
    )
    parser.add_argument(
        "--severity-range",
        type=str,
        default=None,
        help="Severity range for fault detection testing (e.g., '10,20' for early faults). "
             "Format: 'min,max'. Only affects fault test datasets, not training. "
             "Use '10,20' to match paper's early fault detection protocol.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes per rank.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable torch.cuda.amp automatic mixed precision.",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend to use when launched with torchrun.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["l1", "l2"],
        default="l1",
        help="Loss function type: 'l1' (L1/MAE) or 'l2' (MSE). Paper uses L1.",
    )
    # Split mode arguments (primarily for pronto_merged dataset)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["temporal", "segment_shuffle"],
        default="segment_shuffle",
        help="Data split strategy: 'temporal' for sequential splits, "
             "'segment_shuffle' (default) shuffles segments to balance operating conditions.",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=10,
        help="Number of segments for segment_shuffle mode (default: 10).",
    )
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="0.7,0.2,0.1",
        help="Train/val/test ratios as comma-separated values (default: '0.7,0.2,0.1').",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42). Separate from --seed for reproducibility.",
    )
    parser.add_argument(
        "--train-segments",
        type=str,
        default=None,
        help="Explicit segment indices for training (e.g., '0,1,2,5,7,8,9'). Overrides --split-ratios.",
    )
    parser.add_argument(
        "--val-segments",
        type=str,
        default=None,
        help="Explicit segment indices for validation (e.g., '3,6'). Overrides --split-ratios.",
    )
    parser.add_argument(
        "--test-segments",
        type=str,
        default=None,
        help="Explicit segment indices for testing (e.g., '4'). Overrides --split-ratios.",
    )
    # Early stopping arguments
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping based on validation loss.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Number of epochs with no improvement before stopping (default: 20).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement in validation loss to reset patience (default: 1e-4).",
    )
    args = parser.parse_args()
    if args.dataset_key is None:
        parser.error(
            f"--dataset-key is required. Available adapters: {', '.join(available_datasets)}"
        )
    if args.task == "prediction" and args.pred_horizon <= 0:
        parser.error("--pred-horizon must be > 0 when --task prediction is selected.")
    return args


def parse_cuda_devices(cuda_devices: Optional[str]) -> Optional[List[int]]:
    if not cuda_devices:
        return None

    parsed: List[int] = []
    for token in cuda_devices.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        if not stripped.lstrip("-").isdigit():
            raise ValueError(f"Invalid CUDA device index '{token}'. Use comma-separated integers like '0,1'.")
        parsed.append(int(stripped))

    if not parsed:
        raise ValueError("No valid CUDA device indices were parsed from --cuda-devices.")
    return parsed


def init_distributed_mode(backend: str) -> Tuple[bool, int, int, int]:
    if not dist.is_available():
        return False, 0, 1, 0

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed mode requires CUDA to be available.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)

    return True, rank, world_size, local_rank


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def resolve_devices(
    device_flag: str,
    cuda_index: Optional[int],
    cuda_devices: Optional[str],
) -> Tuple[torch.device, Optional[List[int]]]:
    multi_device_ids = parse_cuda_devices(cuda_devices)

    if multi_device_ids is not None:
        if cuda_index is not None:
            raise ValueError("Use either --cuda-device or --cuda-devices, not both.")
        if len(multi_device_ids) > 1:
            raise ValueError(
                "Only single-GPU execution is supported; pass one index (e.g., --cuda-devices 0)."
            )
        cuda_index = multi_device_ids[0]
        multi_device_ids = None

    if device_flag == "cpu":
        if cuda_index is not None:
            raise ValueError("A CUDA device index was provided but device='cpu'.")
        return torch.device("cpu"), None
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if cuda_index is not None:
            if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
                raise ValueError(f"Requested CUDA device {cuda_index}, but only {torch.cuda.device_count()} devices are visible.")
            torch.cuda.set_device(cuda_index)
            return torch.device(f"cuda:{cuda_index}"), None
        return torch.device("cuda"), None
    # auto
    if torch.cuda.is_available():
        if cuda_index is not None:
            if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
                raise ValueError(f"Requested CUDA device {cuda_index}, but only {torch.cuda.device_count()} devices are visible.")
            torch.cuda.set_device(cuda_index)
            return torch.device(f"cuda:{cuda_index}"), None
        return torch.device("cuda"), None
    if cuda_index is not None:
        raise ValueError("CUDA device index specified but CUDA is not available.")
    return torch.device("cpu"), None


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def init_model(
    device: torch.device,
    window_size: int,
    ocvar_dim: int,
    n_nodes: int,
    task: str,
    pred_horizon: int,
    model_args: Optional[argparse.Namespace] = None,
) -> DySTGAT:
    cfg.set_dataset_params(
        n_nodes=n_nodes,
        window_size=window_size,
        ocvar_dim=ocvar_dim,
        pred_horizon=pred_horizon,
        task=task,
    )
    cfg.device = str(device)
    cfg.validate()

    use_spectral = bool(getattr(model_args, "use_spectral_view", False)) if model_args is not None else False
    freq_embed_dim = getattr(model_args, "freq_embed_dim", 16) if model_args is not None else 16
    freq_bins = getattr(model_args, "freq_bins", 0) if model_args is not None else 0
    freq_band_mix = getattr(model_args, "freq_band_mix", "none") if model_args is not None else "none"
    freq_use_log = getattr(model_args, "freq_use_log", True) if model_args is not None else True
    freq_use_spectral_features = (
        getattr(model_args, "freq_use_spectral_features", False) if model_args is not None else False
    )
    freq_topk = getattr(model_args, "freq_topk", None) if model_args is not None else None
    share_gnn_weights = bool(getattr(model_args, "share_gnn_weights", False)) if model_args is not None else False
    fuse_mode = getattr(model_args, "fuse_mode", "concat") if model_args is not None else "concat"
    divergence_type = getattr(model_args, "divergence_type", "js") if model_args is not None else "js"

    model = DySTGAT(
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
        use_spectral_view=use_spectral,
        freq_node_embed_dim=freq_embed_dim,
        freq_max_bins=freq_bins,
        freq_band_mixer=freq_band_mix,
        freq_use_log=freq_use_log,
        freq_use_spectral_features=freq_use_spectral_features,
        freq_topk=freq_topk,
        share_gnn_weights=share_gnn_weights,
        fuse_mode=fuse_mode,
        divergence_type=divergence_type,
        flip_output=(task == "reconstruction"),
        task=task,
        pred_horizon=pred_horizon,
    )
    # Disable cuDNN weight flattening before transferring to device to avoid
    # CUDNN_STATUS_BAD_PARAM in multi-process setups.
    for module in model.modules():
        if isinstance(module, torch.nn.GRU):
            module.flatten_parameters = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    return model.to(device)


def forward_model(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    *,
    return_graph: bool = False,
):
    if isinstance(model, TorchDDP):
        batch_obj = batch.to(device)
        outputs = model(batch_obj, return_graph=return_graph)
        return outputs, batch_obj

    batch_obj = batch.to(device)
    outputs = model(batch_obj, return_graph=return_graph)
    return outputs, batch_obj


def unpack_model_outputs(outputs):
    """
    Normalize model outputs to (recon, edge_index, edge_attr, aux_dict).
    Aux may contain spectral graph and divergence when enabled.
    """
    aux = {}
    edge_index = edge_attr = None
    if isinstance(outputs, tuple):
        if len(outputs) == 4:
            recon, edge_index, edge_attr, aux = outputs
        elif len(outputs) == 3:
            recon, edge_index, edge_attr = outputs
        else:
            recon = outputs[0]
    else:
        recon = outputs
    return recon, edge_index, edge_attr, aux


def resolve_target(batch_obj: Batch, recon: torch.Tensor, task: str) -> torch.Tensor:
    if task == "prediction":
        if not hasattr(batch_obj, "y_future"):
            raise ValueError("Prediction task requires batch.y_future targets.")
        target = batch_obj.y_future
    else:
        target = batch_obj.x
    return target.reshape_as(recon)


def compute_recon_loss(
    model: torch.nn.Module,
    batch,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, Batch]:
    recon, batch_obj = forward_model(model, batch, device, return_graph=False)
    # Sanitize any non-finite values before loss
    if not torch.isfinite(recon).all():
        print("Warning: non-finite reconstruction detected in compute_recon_loss; sanitizing.")
        recon = torch.nan_to_num(recon, nan=0.0, posinf=1e6, neginf=-1e6)
    task = getattr(cfg.dataset, "task", "reconstruction")
    target = resolve_target(batch_obj, recon, task)
    if not torch.isfinite(target).all():
        print("Warning: non-finite target detected in compute_recon_loss; sanitizing.")
        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
    loss = criterion(recon, target)
    return loss, batch_obj


def train_epoch(
    model: DySTGAT,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    distributed: bool = False,
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
) -> Tuple[float, float, float, float, Optional[dict]]:
    model.train()
    base_model = unwrap_model(model)
    task = getattr(cfg.dataset, "task", "reconstruction")
    use_graph = getattr(cfg, "anomaly_weight", 0.0) > 0.0
    div_weight = getattr(cfg, "lambda_div", 0.0)
    running_total_loss = 0.0
    running_recon_loss = 0.0
    running_anom = 0.0
    running_div = 0.0
    sample_count = 0
    # Gate diagnostics for gated fusion mode
    gate_mean_sum = 0.0
    gate_std_sum = 0.0
    z_freq_norm_sum = 0.0
    gate_batch_count = 0

    for raw_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp_enabled):
            if use_graph:
                outputs, batch_obj = forward_model(
                    model, raw_batch, device, return_graph=True
                )
                recon, edge_index, edge_attr, aux = unpack_model_outputs(outputs)
            else:
                recon, batch_obj = forward_model(model, raw_batch, device, return_graph=False)
                edge_index = edge_attr = None
                aux = {}

            if not torch.isfinite(recon).all():
                print("Warning: non-finite reconstruction detected; sanitizing and skipping anomaly/divergence for this batch.")
                recon = torch.nan_to_num(recon, nan=0.0, posinf=1e6, neginf=-1e6)
            target = resolve_target(batch_obj, recon, task)
            if not torch.isfinite(target).all():
                print("Warning: non-finite target detected; sanitizing.")
                target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
            recon_loss = criterion(recon, target)
            anom_score = torch.tensor(0.0, device=device)
            div_loss = aux.get("divergence_loss", torch.tensor(0.0, device=device))
            if use_graph:
                anom_score = base_model.compute_topology_aware_anomaly_score(
                    target, recon, edge_index, edge_attr
                )
                if not torch.isfinite(anom_score):
                    print("Warning: non-finite anomaly score; zeroing for this batch.")
                    anom_score = torch.tensor(0.0, device=device)
            if not torch.isfinite(div_loss).all():
                print("Warning: non-finite divergence loss; zeroing for this batch.")
                div_loss = torch.tensor(0.0, device=device)
            loss = recon_loss + cfg.anomaly_weight * anom_score + div_weight * div_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = batch_obj.num_graphs
        running_total_loss += loss.detach().item() * batch_size
        running_recon_loss += recon_loss.detach().item() * batch_size
        running_anom += anom_score.detach().item() * batch_size
        running_div += div_loss.detach().item() * batch_size
        sample_count += batch_size

        # Track gate diagnostics for gated fusion mode
        if use_graph and "gate_mean" in aux:
            gate_mean_sum += aux["gate_mean"]
            gate_std_sum += aux["gate_std"]
            z_freq_norm_sum += aux["z_freq_norm"]
            gate_batch_count += 1

    totals = torch.tensor(
        [running_total_loss, running_recon_loss, running_anom, running_div, sample_count],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss, total_recon, total_anom, total_div, total_samples = totals.tolist()
    denom = max(total_samples, 1.0)

    # Compute gate diagnostics (only for gated fusion mode)
    gate_diagnostics = None
    if gate_batch_count > 0:
        gate_diagnostics = {
            "gate_mean": gate_mean_sum / gate_batch_count,
            "gate_std": gate_std_sum / gate_batch_count,
            "z_freq_norm": z_freq_norm_sum / gate_batch_count,
        }

    return (
        float(total_loss / denom),
        float(total_recon / denom),
        float(total_anom / denom),
        float(total_div / denom),
        gate_diagnostics,
    )


@torch.no_grad()
def evaluate(
    model: DySTGAT,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    distributed: bool = False,
    amp_enabled: bool = False,
    return_scores: bool = False,
) -> Tuple[float, float, Optional[np.ndarray], float]:
    model.eval()
    base_model = unwrap_model(model)
    task = getattr(cfg.dataset, "task", "reconstruction")
    running_loss = 0.0
    running_score = 0.0
    running_div = 0.0
    sample_count = 0
    all_scores = []

    for raw_batch in loader:
        with autocast("cuda", enabled=amp_enabled):
            outputs, batch_obj = forward_model(model, raw_batch, device, return_graph=True)
            recon, edge_index, edge_attr, aux = unpack_model_outputs(outputs)
            if not torch.isfinite(recon).all():
                print("Warning: non-finite reconstruction detected during evaluation; sanitizing.")
                recon = torch.nan_to_num(recon, nan=0.0, posinf=1e6, neginf=-1e6)
            target = resolve_target(batch_obj, recon, task)
            if not torch.isfinite(target).all():
                print("Warning: non-finite target detected during evaluation; sanitizing.")
                target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
            loss = criterion(recon, target)

        # Compute aggregate score for metrics
        score = base_model.compute_topology_aware_anomaly_score(
            target, recon, edge_index, edge_attr
        )
        div_loss = aux.get("divergence_loss", torch.tensor(0.0, device=device))

        # If detailed scores requested, compute per-sample scores
        if return_scores:
            # shape: [batch_size]
            batch_scores = base_model.compute_anomaly_scores_per_sample(
                target, recon, edge_index, edge_attr
            )
            all_scores.append(batch_scores.cpu().numpy())

        batch_size = batch_obj.num_graphs
        running_loss += loss.detach().item() * batch_size
        running_score += score.item() * batch_size
        running_div += div_loss.detach().item() * batch_size
        sample_count += batch_size

    totals = torch.tensor(
        [running_loss, running_score, running_div, sample_count],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss, total_score, total_div, total_samples = totals.tolist()
    denom = max(total_samples, 1.0)
    
    scores_array = None
    if return_scores and all_scores:
        scores_array = np.concatenate(all_scores)

    return float(total_loss / denom), float(total_score / denom), scores_array, float(total_div / denom)


@torch.no_grad()
def evaluate_with_per_sample_metrics(
    model: DySTGAT,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    distributed: bool = False,
    amp_enabled: bool = False,
) -> Tuple[float, float, float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    base_model = unwrap_model(model)
    task = getattr(cfg.dataset, "task", "reconstruction")
    running_loss = 0.0
    running_score = 0.0
    running_div = 0.0
    sample_count = 0
    all_anom = []
    all_mse = []
    all_div = []

    for raw_batch in loader:
        with autocast("cuda", enabled=amp_enabled):
            outputs, batch_obj = forward_model(model, raw_batch, device, return_graph=True)
            recon, edge_index, edge_attr, aux = unpack_model_outputs(outputs)
            if not torch.isfinite(recon).all():
                recon = torch.nan_to_num(recon, nan=0.0, posinf=1e6, neginf=-1e6)
            target = resolve_target(batch_obj, recon, task)
            if not torch.isfinite(target).all():
                target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
            loss = criterion(recon, target)

        score = base_model.compute_topology_aware_anomaly_score(
            target, recon, edge_index, edge_attr
        )
        div_loss = aux.get("divergence_loss", torch.tensor(0.0, device=device))

        anom_per = base_model.compute_anomaly_scores_per_sample(
            target, recon, edge_index, edge_attr
        )
        anom_per = torch.nan_to_num(anom_per, nan=0.0, posinf=1e6, neginf=-1e6)

        n = cfg.dataset.n_nodes
        b = max(int(target.shape[0] // n), 1)
        err = (target - recon) ** 2
        mse_per = err.view(b, n, -1).mean(dim=(1, 2))

        div_per = aux.get("divergence_score", None)
        if div_per is None:
            div_per = torch.zeros_like(anom_per)
        else:
            div_per = div_per.view(-1)
            if div_per.numel() != anom_per.numel():
                if div_per.numel() > anom_per.numel():
                    div_per = div_per[: anom_per.numel()]
                else:
                    pad = torch.zeros(anom_per.numel() - div_per.numel(), device=div_per.device)
                    div_per = torch.cat([div_per, pad])

        all_anom.append(anom_per.detach().cpu().numpy())
        all_mse.append(mse_per.detach().cpu().numpy())
        all_div.append(div_per.detach().cpu().numpy())

        batch_size = batch_obj.num_graphs
        running_loss += loss.detach().item() * batch_size
        running_score += score.detach().item() * batch_size
        running_div += div_loss.detach().item() * batch_size
        sample_count += batch_size

    totals = torch.tensor(
        [running_loss, running_score, running_div, sample_count],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss, total_score, total_div, total_samples = totals.tolist()
    denom = max(total_samples, 1.0)

    anom_array = np.concatenate(all_anom) if all_anom else None
    mse_array = np.concatenate(all_mse) if all_mse else None
    div_array = np.concatenate(all_div) if all_div else None

    return (
        float(total_loss / denom),
        float(total_score / denom),
        float(total_div / denom),
        anom_array,
        mse_array,
        div_array,
    )


@torch.no_grad()
def evaluate_tests_and_plot(
    model: DySTGAT,
    loaders: Dict[str, torch.utils.data.DataLoader],
    criterion: torch.nn.Module,
    device: torch.device,
    output_dir: str,
    *,
    distributed: bool = False,
    amp_enabled: bool = False,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating Plotly test metrics (anomaly, MSE, divergence)...")

    results_anom: Dict[str, np.ndarray] = {}
    results_mse: Dict[str, np.ndarray] = {}
    results_div: Dict[str, np.ndarray] = {}
    # Separate validation and test baseline scores to prevent data leakage
    # Per paper (p.11): threshold should be based on 95th percentile of normal VALIDATION set
    val_baseline_scores = None  # For threshold computation (no leakage)
    test_baseline_scores = None  # For AUC computation (test normal vs test fault)

    # Collect per-sample metrics for each test loader
    for name, loader in loaders.items():
        loss, score, div_loss, anom_arr, mse_arr, div_arr = evaluate_with_per_sample_metrics(
            model,
            loader,
            criterion,
            device,
            distributed=distributed,
            amp_enabled=amp_enabled,
        )
        metrics[name] = {"recon_loss": loss, "anomaly_score": score, "divergence": div_loss}

        if anom_arr is not None:
            results_anom[name] = anom_arr
            results_mse[name] = mse_arr if mse_arr is not None else np.zeros_like(anom_arr)
            results_div[name] = div_arr if div_arr is not None else np.zeros_like(anom_arr)

            # Identify baseline scores - separate val from test to prevent threshold leakage
            name_lower = name.lower()
            if "baseline" in name_lower or "fault_free" in name_lower:
                # Check if this is validation or test baseline
                if "val" in name_lower:
                    # Validation baseline - used for threshold computation
                    if val_baseline_scores is None:
                        val_baseline_scores = anom_arr
                    else:
                        val_baseline_scores = np.concatenate([val_baseline_scores, anom_arr])
                else:
                    # Test baseline - used for AUC computation
                    if test_baseline_scores is None:
                        test_baseline_scores = anom_arr
                    else:
                        test_baseline_scores = np.concatenate([test_baseline_scores, anom_arr])

    # Save raw scores
    if results_anom:
        np.savez(os.path.join(output_dir, "anomaly_scores.npz"), **results_anom)
        np.savez(os.path.join(output_dir, "mse_scores.npz"), **results_mse)
        np.savez(os.path.join(output_dir, "divergence_scores.npz"), **results_div)

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except Exception:
            print("Plotly not available; skipping HTML plot generation.")
        else:
            def sanitize_plot_name(value: str) -> str:
                return (
                    value.replace("/", "_")
                    .replace("\\", "_")
                    .replace(" ", "_")
                    .replace(":", "_")
                    .replace(",", "_")
                )

            for name, series in results_anom.items():
                x = np.arange(series.shape[0])
                fig = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    subplot_titles=("Anomaly Score", "MSE", "Divergence"),
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=series,
                        mode="lines",
                        name="anomaly",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=results_mse[name],
                        mode="lines",
                        name="mse",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=results_div[name],
                        mode="lines",
                        name="divergence",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

                fig.update_layout(
                    height=900,
                    width=1200,
                    title_text=f"Test Set Metrics - {name}",
                )
                fig.update_xaxes(title_text="Sample Index", row=3, col=1)
                fig.update_yaxes(title_text="Anomaly Score", row=1, col=1)
                fig.update_yaxes(title_text="MSE", row=2, col=1)
                fig.update_yaxes(title_text="Divergence", row=3, col=1)
                fig.write_html(
                    os.path.join(output_dir, f"test_metrics_plot_{sanitize_plot_name(name)}.html")
                )
    
    # Second pass: Compute Classification Metrics (AUC, Precision, Recall, F1 variants, Delay, Ambiguity)
    # Only if we have baselines to compare against
    # Require val_baseline for threshold; test_baseline for AUC (fall back to val if no test)
    if val_baseline_scores is not None:
        # Use test baseline for AUC if available, otherwise fall back to val baseline
        auc_baseline_scores = test_baseline_scores if test_baseline_scores is not None else val_baseline_scores
        detailed_metrics_path = os.path.join(output_dir, "detailed_test_metrics.csv")
        print(f"Calculating AUC/F1 metrics...")
        print(f"  Threshold computed from VALIDATION baseline (N={len(val_baseline_scores)}) - no leakage")
        print(f"  AUC computed using TEST baseline (N={len(auc_baseline_scores)})")

        # Score distribution debug output
        print(f"\n=== Score Distribution Debug ===")
        print(f"Baseline: min={np.min(auc_baseline_scores):.4f}, max={np.max(auc_baseline_scores):.4f}, "
              f"mean={np.mean(auc_baseline_scores):.4f}, p95={np.percentile(auc_baseline_scores, 95):.4f}")
        for name, scores in results_anom.items():
            if "baseline" not in name.lower() and "fault_free" not in name.lower():
                print(f"{name}: min={np.min(scores):.4f}, max={np.max(scores):.4f}, "
                      f"mean={np.mean(scores):.4f}, p5={np.percentile(scores, 5):.4f}")
        print(f"================================\n")
        
        with open(detailed_metrics_path, 'w', newline='') as csvfile:
            fieldnames = [
                'test_set',
                'auc_roc',
                'precision',
                'recall',
                'f1_score',          # F1 at fixed threshold (95th percentile)
                'best_f1',           # F1* (max over PR curve)
                'ambiguity',         # 1 - 2|AUC-0.5|
                'threshold',         # fixed threshold (95th percentile of baseline)
                'best_threshold',    # threshold achieving best_f1
                'delay_best_thr',    # detection delay (samples) at best_threshold
                # TEA (Temporal Evidence Accumulation) metrics
                'tea_auc',           # AUC with TEA post-processing
                'tea_best_f1',       # Best F1 with TEA
                'tea_best_window',   # Window size that achieved best TEA AUC
                'tea_auc_delta',     # Improvement in AUC from TEA
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for name, scores in results_anom.items():
                # Skip if it IS the baseline
                if "baseline" in name.lower() or "fault_free" in name.lower():
                    continue

                # Construct labels for AUC computation
                # Use test baseline (or fallback) = 0, This Fault = 1
                y_true = np.concatenate([np.zeros(len(auc_baseline_scores)), np.ones(len(scores))])
                y_scores = np.concatenate([auc_baseline_scores, scores])

                try:
                    auc = roc_auc_score(y_true, y_scores)
                except ValueError:
                    auc = 0.0

                # Fixed threshold for F1: 95th percentile of VALIDATION baseline (per paper p.11)
                # This prevents data leakage - threshold is set without seeing test data
                threshold = np.percentile(val_baseline_scores, 95)
                y_pred = (y_scores > threshold).astype(int)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )

                # Best F1 (F1*): scan PR curve
                pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_scores)
                # Exclude last element (endpoint where recall=0, giving F1=0)
                f1_curve = 2 * pr_prec[:-1] * pr_rec[:-1] / (pr_prec[:-1] + pr_rec[:-1] + 1e-12)
                best_idx = int(np.argmax(f1_curve))
                best_f1 = float(f1_curve[best_idx])
                # precision_recall_curve returns thresholds len = len(prec)-1
                best_threshold = float(pr_thresh[best_idx]) if best_idx < len(pr_thresh) else float(pr_thresh[-1])

                # Detection delay at best_threshold: samples until first detection in fault segment
                # Using fault scores only (all ones) since baseline prepends zeros.
                fault_scores = scores
                delay_idx = next(
                    (i for i, s in enumerate(fault_scores) if s > best_threshold),
                    len(fault_scores),
                )
                delay_best = int(delay_idx)

                # Ambiguity (for Pronto novel OCs; harmless for others)
                ambiguity = 1.0 - 2.0 * abs(auc - 0.5)

                # TEA (Temporal Evidence Accumulation) for incipient fault detection
                # Window sizes: 300 (~5min), 600 (~10min), 1800 (~30min) at 1s sampling
                # PRONTO raw CSV uses 1s intervals (per header: "Interval,1s")
                # Use test baseline for TEA AUC computation (consistent with raw AUC)
                tea_metrics = compute_tea_metrics(
                    auc_baseline_scores,
                    scores,
                    window_sizes=[300, 600, 1800],
                    return_best_window=True,
                )
                tea_auc = tea_metrics['auc']
                tea_best_f1 = tea_metrics['best_f1']
                tea_best_window = tea_metrics['best_window']
                tea_auc_delta = tea_auc - auc

                row = {
                    'test_set': name,
                    'auc_roc': f"{auc:.4f}",
                    'precision': f"{prec:.4f}",
                    'recall': f"{rec:.4f}",
                    'f1_score': f"{f1:.4f}",
                    'best_f1': f"{best_f1:.4f}",
                    'ambiguity': f"{ambiguity:.4f}",
                    'threshold': f"{threshold:.6f}",
                    'best_threshold': f"{best_threshold:.6f}",
                    'delay_best_thr': delay_best,
                    'tea_auc': f"{tea_auc:.4f}",
                    'tea_best_f1': f"{tea_best_f1:.4f}",
                    'tea_best_window': tea_best_window,
                    'tea_auc_delta': f"{tea_auc_delta:+.4f}",
                }
                writer.writerow(row)
                
                # Update the returned metrics dict for printing
                metrics[name]['auc'] = auc
                metrics[name]['f1'] = f1
                metrics[name]['best_f1'] = best_f1
                metrics[name]['ambiguity'] = ambiguity
                metrics[name]['delay_best_thr'] = delay_best
                metrics[name]['tea_auc'] = tea_auc
                metrics[name]['tea_best_f1'] = tea_best_f1
                metrics[name]['tea_auc_delta'] = tea_auc_delta

        print(f"Detailed metrics saved to {detailed_metrics_path}")

    print(f"Plots and scores saved to {output_dir}")
            
    return metrics


def main() -> None:
    args = parse_args()
    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval-only requires --checkpoint to be specified.")

    # Set random seed for reproducibility if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Force single-GPU / single-process execution. Any torchrun/DDP environment
    # variables are intentionally ignored.
    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    is_main_process = True

    try:
        device, _ = resolve_devices(args.device, args.cuda_device, args.cuda_devices)
        if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
            print("Single-GPU mode enforced; ignoring torchrun/Distributed environment variables.")

        if device.type == "cuda" and args.seed is None:
            # Only enable benchmark when not requiring reproducibility
            torch.backends.cudnn.benchmark = True
        cfg.anomaly_weight = float(args.anomaly_weight)
        cfg.lambda_div = float(args.lambda_div)

        adapter = get_adapter(args.dataset_key)
        adapter.ensure("training")
        data_dir = args.data_dir or adapter.get_default_data_dir()
        if data_dir is None:
            raise ValueError(
                f"Dataset adapter '{args.dataset_key}' does not define a default data directory; "
                "please supply --data-dir."
            )

        feature_option = args.ashrae_feature_option if args.dataset_key == "ashrae" else None
        ashrae_faults: Optional[List[str]] = None
        if args.dataset_key == "ashrae" and args.ashrae_faults.lower() != "all":
            ashrae_faults = [f.strip() for f in args.ashrae_faults.split(",") if f.strip()]
        if args.skip_test and args.dataset_key == "ashrae":
            ashrae_faults = []

        control_var_names = adapter.get_control_variables(data_dir, feature_option=feature_option)
        measurement_var_names = adapter.get_measurement_variables(feature_option)
        model = init_model(
            device,
            args.window_size,
            len(control_var_names),
            adapter.measurement_count(feature_option),
            args.task,
            args.pred_horizon,
            model_args=args,
        )

        checkpoint_root = args.checkpoint_dir or os.path.join("checkpoints", adapter.key)
        checkpoint_root_path = Path(checkpoint_root).expanduser().resolve()
        if args.save_model:
            save_model_path: Optional[Path] = Path(args.save_model).expanduser().resolve()
        elif not args.eval_only:
            save_model_path = checkpoint_root_path / f"dystgat_{adapter.key}_best.pt"
        else:
            save_model_path = None
        if save_model_path is not None:
            save_model_path.parent.mkdir(parents=True, exist_ok=True)

        if is_main_process:
            print("=" * 80)
            mode_desc = "Evaluating" if args.eval_only else "Training"
            print(f"{mode_desc} DySTGAT ({args.dataset_key}) on device: {device}")
            print(f"Data directory: {data_dir}")
            if not args.eval_only:
                print(f"Checkpoint root: {checkpoint_root_path}")
            if save_model_path is not None:
                print(f"Final model will be saved to: {save_model_path.as_posix()}")
            print("\nSelected feature columns:")
            print(f"  Controls (U): {control_var_names}")
            print(f"  Measurements (X): {measurement_var_names}")
            print("  Labels and timestamps are excluded from model inputs; timestamps are only for ordering, labels for filtering/metrics.")
            print(f"  Task: {args.task} (pred_horizon={args.pred_horizon})")
            print("=" * 80)

        if args.checkpoint:
            if not os.path.isfile(args.checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            load_result = model.load_state_dict(checkpoint, strict=False)
            if is_main_process:
                print(f"Loaded checkpoint from {args.checkpoint}")
                if load_result.missing_keys:
                    print(f"  Missing keys: {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    print(f"  Unexpected keys: {load_result.unexpected_keys}")

        base_model = unwrap_model(model)
        # Select loss function based on --loss-type (paper uses L1)
        if args.loss_type == "l1":
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        scaler = GradScaler("cuda") if args.use_amp and device.type == "cuda" else None
        amp_enabled = scaler is not None

        effective_test_stride = args.test_stride if args.test_stride is not None else args.val_stride

        # Parse severity range if provided
        severity_range = None
        if args.severity_range:
            try:
                min_sev, max_sev = map(int, args.severity_range.split(','))
                severity_range = (min_sev, max_sev)
                if is_main_process:
                    print(f"🎯 Filtering fault test data to severity range: [{min_sev}, {max_sev}]")
                    if min_sev <= 20:
                        print("   → Testing EARLY fault detection (matching paper's protocol)")
                    elif min_sev <= 40:
                        print("   → Testing MODERATE fault detection")
                    else:
                        print("   → Testing SEVERE fault detection (easier)")
            except ValueError:
                raise ValueError(
                    f"Invalid --severity-range format: '{args.severity_range}'. "
                    "Expected 'min,max' (e.g., '10,20')"
                )

        # Parse split ratios if provided
        split_ratios = tuple(float(x) for x in args.split_ratios.split(','))
        if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 0.01:
            raise ValueError(
                f"Invalid --split-ratios: '{args.split_ratios}'. "
                "Expected 3 values that sum to 1.0 (e.g., '0.7,0.2,0.1')"
            )

        # Parse explicit segment assignments if provided
        train_segments = None
        val_segments = None
        test_segments = None
        if args.train_segments and args.val_segments and args.test_segments:
            train_segments = [int(x) for x in args.train_segments.split(',')]
            val_segments = [int(x) for x in args.val_segments.split(',')]
            test_segments = [int(x) for x in args.test_segments.split(',')]
            if is_main_process:
                print(f"Using explicit segment assignment:")
                print(f"  Train: {train_segments}")
                print(f"  Val: {val_segments}")
                print(f"  Test: {test_segments}")

        train_loader, val_loader, test_loaders = adapter.create_dataloaders(
            window_size=cfg.dataset.window_size,
            batch_size=args.batch_size,
            train_stride=args.train_stride,
            val_stride=args.val_stride,
            test_stride=effective_test_stride,
            data_dir=data_dir,
            num_workers=args.num_workers,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            baseline_from=args.baseline_from,
            severity_range=severity_range,
            feature_option=feature_option,
            fault_keys=ashrae_faults,
            pred_horizon=args.pred_horizon,
            # Split mode parameters (used by pronto_merged dataset)
            split_mode=args.split_mode,
            n_segments=args.n_segments,
            split_ratios=split_ratios,
            random_seed=args.data_seed,
            train_segments=train_segments,
            val_segments=val_segments,
            test_segments=test_segments,
        )

        best_val_loss = float("inf")
        best_val_anom = float("inf")
        best_state = None
        checkpoint_manager = None

        # Early stopping state
        patience_counter = 0
        early_stop_triggered = False

        if not args.eval_only:
            checkpoint_manager = EpochCheckpointManager(
                str(checkpoint_root_path), prefix=f"dystgat_{args.dataset_key}"
            )
            if is_main_process:
                print(f"\nSaving per-epoch checkpoints to: {checkpoint_manager.run_path}")
                command_path = Path(checkpoint_manager.run_path) / "train_command.txt"
                command = shlex.join([sys.executable] + sys.argv)
                command_path.write_text(f"{command}\n", encoding="utf-8")

            for epoch in range(1, args.epochs + 1):
                start_time = time.time()
                train_total_loss, train_recon_loss, train_anom, train_div, gate_diag = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    distributed=distributed,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                )
                val_loss, val_score, _, val_div = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    distributed=distributed,
                    amp_enabled=amp_enabled,
                    return_scores=False,
                )
                elapsed = time.time() - start_time

                if is_main_process:
                    log_msg = (
                        f"[Epoch {epoch:03d}] train_total={train_total_loss:.6f} "
                        f"train_recon={train_recon_loss:.6f} train_anom={train_anom:.6f} train_div={train_div:.6f} "
                        f"val_loss={val_loss:.6f} val_anom={val_score:.6f} val_div={val_div:.6f} "
                        f"time={elapsed:.1f}s"
                    )
                    print(log_msg)
                    # Gate diagnostics for monitoring spectral branch contribution (gated fusion only)
                    if gate_diag is not None:
                        gate_mean = gate_diag["gate_mean"]
                        gate_std = gate_diag["gate_std"]
                        z_freq_norm = gate_diag["z_freq_norm"]
                        # Warn if gate is extreme (spectral branch may be underutilized)
                        gate_warning = ""
                        if gate_mean > 0.9:
                            gate_warning = " [!] spectral underutilized"
                        elif gate_mean < 0.1:
                            gate_warning = " [!] temporal underutilized"
                        print(
                            f"  ↳ gate: mean={gate_mean:.3f} std={gate_std:.3f} "
                            f"z_freq_norm={z_freq_norm:.2f}{gate_warning}"
                        )
                    if checkpoint_manager is not None:
                        train_rmse = math.sqrt(train_recon_loss)
                        val_rmse = math.sqrt(val_loss)
                        extra_state = {
                            "train_mse": f"{train_recon_loss:.6f}",
                            "train_rmse": f"{train_rmse:.6f}",
                            "train_div": f"{train_div:.6f}",
                            "train_anom": f"{train_anom:.6f}",
                            "val_mse": f"{val_loss:.6f}",
                            "val_rmse": f"{val_rmse:.6f}",
                            "val_div": f"{val_div:.6f}",
                            "val_anom": f"{val_score:.6f}",
                        }
                        # Include gate diagnostics in checkpoint (gated fusion only)
                        if gate_diag is not None:
                            extra_state["gate_mean"] = f"{gate_diag['gate_mean']:.3f}"
                            extra_state["gate_std"] = f"{gate_diag['gate_std']:.3f}"
                            extra_state["z_freq_norm"] = f"{gate_diag['z_freq_norm']:.2f}"
                        checkpoint_path = checkpoint_manager.save_epoch(
                            epoch=epoch,
                            model=base_model,
                            train_loss=train_total_loss,
                            val_loss=val_loss,
                            val_anom=val_score,
                            elapsed_time=elapsed,
                            extra_state=extra_state,
                        )
                        print(f"  ↳ checkpoint saved: {checkpoint_path.name}")

                improved = False
                # Use a slightly more robust improvement check
                if val_score < best_val_anom - 1e-9:
                    improved = True
                elif abs(val_score - best_val_anom) <= 1e-9 and val_loss < best_val_loss:
                    improved = True

                if improved:
                    best_val_anom = val_score
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
                    patience_counter = 0  # Reset early stopping counter
                    if is_main_process:
                        print(
                            f"  ↳ new best model (val_anom={best_val_anom:.6f}, val_loss={best_val_loss:.6f})"
                        )
                else:
                    # Early stopping check
                    if args.early_stopping:
                        patience_counter += 1
                        if is_main_process and patience_counter > 0:
                            print(f"  ↳ no improvement for {patience_counter}/{args.patience} epochs")
                        if patience_counter >= args.patience:
                            if is_main_process:
                                print(f"\n⏹ Early stopping triggered at epoch {epoch} (no improvement for {args.patience} epochs)")
                            early_stop_triggered = True
                            break

            if best_state is not None:
                base_model.load_state_dict(best_state)
                if is_main_process:
                    print(f"\nBest validation metrics: loss={best_val_loss:.6f}, anomaly={best_val_anom:.6f}")
            elif is_main_process:
                print("\nWarning: No improvement over initial epoch.")

            if is_main_process and save_model_path is not None:
                state_to_save = best_state if best_state is not None else base_model.state_dict()
                torch.save(state_to_save, save_model_path)
                print(f"\nSaved model checkpoint to: {save_model_path.as_posix()}")
        else:
            if is_main_process:
                print("\nEvaluation-only mode: skipping training loop.")

        val_summary_loss, val_summary_score, val_scores_array, val_summary_div = evaluate(
            model,
            val_loader,
            criterion,
            device,
            distributed=distributed,
            amp_enabled=amp_enabled,
            return_scores=True,
        )

        if is_main_process:
            print(
                f"\nValidation summary -> recon_loss={val_summary_loss:.6f} "
                f"anomaly_score={val_summary_score:.6f} div={val_summary_div:.6f}"
            )

        if args.skip_test:
            if is_main_process:
                print("\nSkipping test evaluation (--skip-test).")
            return
            
        if is_main_process:
            # Create output directory for plots
            if checkpoint_manager is not None:
                plot_dir = os.path.join(checkpoint_manager.run_path, "plots")
            else:
                # Fallback for eval-only mode
                plot_dir = os.path.join(checkpoint_root_path, f"eval_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
            os.makedirs(plot_dir, exist_ok=True)
            
            # Save validation plots first
            if val_scores_array is not None:
                plt.figure(figsize=(12, 6))
                plt.plot(val_scores_array, label=f'Validation (Avg: {val_summary_score:.4f})', color='green')
                plt.title("Anomaly Scores over Time - Validation Set")
                plt.xlabel("Sample Index")
                plt.ylabel("Anomaly Score")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plot_dir, "anomaly_plot_validation.png"))
                plt.close()

            print("\nEvaluating on benchmark validation/test plus fault datasets...")

            eval_loaders = {"baseline_val": val_loader}
            for name, loader in test_loaders.items():
                key = "baseline_test" if name == "baseline" else name
                eval_loaders[key] = loader

            test_scores = evaluate_tests_and_plot(
                model,
                eval_loaders,
                criterion,
                device,
                output_dir=plot_dir,
                distributed=distributed,
                amp_enabled=amp_enabled,
            )

            for name, metrics in test_scores.items():
                auc_str = f" auc={metrics['auc']:.4f}" if 'auc' in metrics else ""
                f1_str = f" f1={metrics['f1']:.4f}" if 'f1' in metrics else ""
                div_str = f" div={metrics['divergence']:.6f}" if 'divergence' in metrics else ""
                tea_str = f" tea_auc={metrics['tea_auc']:.4f}({metrics['tea_auc_delta']:+.4f})" if 'tea_auc' in metrics else ""
                print(
                    f"  {name:30s}: recon_loss={metrics['recon_loss']:.6f} "
                    f"anomaly_score={metrics['anomaly_score']:.6f}{div_str}{auc_str}{f1_str}{tea_str}"
                )

            if args.eval_only:
                print("\nEvaluation complete.")
            else:
                print("\nTraining complete.")
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    from datetime import datetime # Import locally to avoid top-level shadowing if needed
    main()
