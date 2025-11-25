
import argparse
import os
import sys
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as TorchDDP
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dyedgegat"))

from src.config import cfg
from datasets import get_adapter, list_adapter_keys
from src.model.dyedgegat import DyEdgeGAT
from src.utils.checkpoint import EpochCheckpointManager


def parse_args() -> argparse.Namespace:
    available_datasets = list_adapter_keys()
    parser = argparse.ArgumentParser(description="Train DyEdgeGAT on refrigeration dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
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
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument(
        "--dataset-key",
        type=str,
        choices=available_datasets,
        help=f"Dataset adapter to use. Available: {', '.join(available_datasets)}.",
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
        choices=["val", "train"],
        default="val",
        help="Source split for the baseline (normal) test loader; default reuses validation files.",
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
    args = parser.parse_args()
    if args.dataset_key is None:
        parser.error(
            f"--dataset-key is required. Available adapters: {', '.join(available_datasets)}"
        )
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
    model_args: Optional[argparse.Namespace] = None,
) -> DyEdgeGAT:
    cfg.set_dataset_params(
        n_nodes=n_nodes,
        window_size=window_size,
        ocvar_dim=ocvar_dim,
    )
    cfg.device = str(device)
    cfg.validate()

    use_spectral = bool(getattr(model_args, "use_spectral_view", False)) if model_args is not None else False
    freq_embed_dim = getattr(model_args, "freq_embed_dim", 16) if model_args is not None else 16
    freq_bins = getattr(model_args, "freq_bins", 0) if model_args is not None else 0
    freq_band_mix = getattr(model_args, "freq_band_mix", "none") if model_args is not None else "none"
    freq_topk = getattr(model_args, "freq_topk", None) if model_args is not None else None
    share_gnn_weights = bool(getattr(model_args, "share_gnn_weights", False)) if model_args is not None else False
    fuse_mode = getattr(model_args, "fuse_mode", "concat") if model_args is not None else "concat"
    divergence_type = getattr(model_args, "divergence_type", "js") if model_args is not None else "js"

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
        use_spectral_view=use_spectral,
        freq_node_embed_dim=freq_embed_dim,
        freq_max_bins=freq_bins,
        freq_band_mixer=freq_band_mix,
        freq_topk=freq_topk,
        share_gnn_weights=share_gnn_weights,
        fuse_mode=fuse_mode,
        divergence_type=divergence_type,
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


def compute_recon_loss(
    model: torch.nn.Module,
    batch,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, Batch]:
    recon, batch_obj = forward_model(model, batch, device, return_graph=False)
    target = batch_obj.x.unsqueeze(-1)
    loss = criterion(recon, target)
    return loss, batch_obj


def train_epoch(
    model: DyEdgeGAT,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    distributed: bool = False,
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
) -> Tuple[float, float, float]:
    model.train()
    base_model = unwrap_model(model)
    use_graph = getattr(cfg, "anomaly_weight", 0.0) > 0.0
    div_weight = getattr(cfg, "lambda_div", 0.0)
    running_total_loss = 0.0
    running_recon_loss = 0.0
    running_anom = 0.0
    running_div = 0.0
    sample_count = 0

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

            target = batch_obj.x.unsqueeze(-1)
            recon_loss = criterion(recon, target)
            anom_score = torch.tensor(0.0, device=device)
            div_loss = aux.get("divergence_loss", torch.tensor(0.0, device=device))
            if use_graph:
                anom_score = base_model.compute_topology_aware_anomaly_score(
                    target, recon, edge_index, edge_attr
                )
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

    totals = torch.tensor(
        [running_total_loss, running_recon_loss, running_anom, running_div, sample_count],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss, total_recon, total_anom, total_div, total_samples = totals.tolist()
    denom = max(total_samples, 1.0)
    return (
        float(total_loss / denom),
        float(total_recon / denom),
        float(total_anom / denom),
        float(total_div / denom),
    )


@torch.no_grad()
def evaluate(
    model: DyEdgeGAT,
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
    running_loss = 0.0
    running_score = 0.0
    running_div = 0.0
    sample_count = 0
    all_scores = []

    for raw_batch in loader:
        with autocast("cuda", enabled=amp_enabled):
            outputs, batch_obj = forward_model(model, raw_batch, device, return_graph=True)
            recon, edge_index, edge_attr, aux = unpack_model_outputs(outputs)
            target = batch_obj.x.unsqueeze(-1)
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
def evaluate_tests_and_plot(
    model: DyEdgeGAT,
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
    
    print("\nGenerating anomaly score plots and detailed metrics...")
    
    # Store all series
    results_dict = {}
    baseline_scores = None
    
    # First pass: Collecting scores
    for name, loader in loaders.items():
        loss, score, scores_array, div_loss = evaluate(
            model,
            loader,
            criterion,
            device,
            distributed=distributed,
            amp_enabled=amp_enabled,
            return_scores=True,
        )
        metrics[name] = {"recon_loss": loss, "anomaly_score": score, "divergence": div_loss}
        
        if scores_array is not None:
            results_dict[name] = scores_array
            
            # Identify baseline scores for AUC calculation
            # Convention: 'baseline' or 'fault_free' in name implies Label 0
            if "baseline" in name.lower() or "fault_free" in name.lower():
                if baseline_scores is None:
                    baseline_scores = scores_array
                else:
                    # Concat if multiple baselines
                    baseline_scores = np.concatenate([baseline_scores, scores_array])

            # Plot individual series
            plt.figure(figsize=(12, 6))
            plt.plot(scores_array, label=f'{name} (Avg: {score:.4f})')
            plt.title(f"Anomaly Scores over Time - {name}")
            plt.xlabel("Sample Index")
            plt.ylabel("Anomaly Score")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"anomaly_plot_{name}.png"))
            plt.close()
            
    # Save raw scores
    np.savez(os.path.join(output_dir, "anomaly_scores.npz"), **results_dict)
    
    # Second pass: Compute Classification Metrics (AUC, Precision, Recall, F1)
    # Only if we have a baseline to compare against
    if baseline_scores is not None:
        detailed_metrics_path = os.path.join(output_dir, "detailed_test_metrics.csv")
        print(f"Calculating AUC/F1 metrics (using baseline N={len(baseline_scores)})...")
        
        with open(detailed_metrics_path, 'w', newline='') as csvfile:
            fieldnames = ['test_set', 'auc_roc', 'precision', 'recall', 'f1_score', 'threshold']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for name, scores in results_dict.items():
                # Skip if it IS the baseline
                if "baseline" in name.lower() or "fault_free" in name.lower():
                    continue
                    
                # Construct labels
                # Baseline = 0, This Fault = 1
                y_true = np.concatenate([np.zeros(len(baseline_scores)), np.ones(len(scores))])
                y_scores = np.concatenate([baseline_scores, scores])
                
                try:
                    auc = roc_auc_score(y_true, y_scores)
                except ValueError:
                    auc = 0.0
                
                # Determine threshold for F1 (simple approach: 95th percentile of baseline)
                # A robust threshold strategy usually requires a separate validation set
                threshold = np.percentile(baseline_scores, 99) # 1% false alarm rate target
                y_pred = (y_scores > threshold).astype(int)
                
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                
                row = {
                    'test_set': name,
                    'auc_roc': f"{auc:.4f}",
                    'precision': f"{prec:.4f}",
                    'recall': f"{rec:.4f}",
                    'f1_score': f"{f1:.4f}",
                    'threshold': f"{threshold:.6f}"
                }
                writer.writerow(row)
                
                # Update the returned metrics dict for printing
                metrics[name]['auc'] = auc
                metrics[name]['f1'] = f1

        print(f"Detailed metrics saved to {detailed_metrics_path}")

    print(f"Plots and scores saved to {output_dir}")
            
    return metrics


def main() -> None:
    args = parse_args()
    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval-only requires --checkpoint to be specified.")
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

        if device.type == "cuda":
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

        control_var_names = adapter.get_control_variables(data_dir)
        model = init_model(
            device,
            args.window_size,
            len(control_var_names),
            adapter.measurement_count(),
            model_args=args,
        )

        checkpoint_root = args.checkpoint_dir or os.path.join("checkpoints", adapter.key)
        checkpoint_root_path = Path(checkpoint_root).expanduser().resolve()
        if args.save_model:
            save_model_path: Optional[Path] = Path(args.save_model).expanduser().resolve()
        elif not args.eval_only:
            save_model_path = checkpoint_root_path / f"dyedgegat_{adapter.key}_best.pt"
        else:
            save_model_path = None
        if save_model_path is not None:
            save_model_path.parent.mkdir(parents=True, exist_ok=True)

        if is_main_process:
            print("=" * 80)
            mode_desc = "Evaluating" if args.eval_only else "Training"
            print(f"{mode_desc} DyEdgeGAT ({args.dataset_key}) on device: {device}")
            print(f"Data directory: {data_dir}")
            if not args.eval_only:
                print(f"Checkpoint root: {checkpoint_root_path}")
            if save_model_path is not None:
                print(f"Final model will be saved to: {save_model_path.as_posix()}")
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
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        scaler = GradScaler("cuda") if args.use_amp and device.type == "cuda" else None
        amp_enabled = scaler is not None

        effective_test_stride = args.test_stride if args.test_stride is not None else args.val_stride

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
        )

        best_val_loss = float("inf")
        best_val_anom = float("inf")
        best_state = None
        checkpoint_manager = None

        if not args.eval_only:
            checkpoint_manager = EpochCheckpointManager(
                str(checkpoint_root_path), prefix=f"dyedgegat_{args.dataset_key}"
            )
            if is_main_process:
                print(f"\nSaving per-epoch checkpoints to: {checkpoint_manager.run_path}")

            for epoch in range(1, args.epochs + 1):
                start_time = time.time()
                train_total_loss, train_recon_loss, train_anom, train_div = train_epoch(
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
                    print(
                        f"[Epoch {epoch:03d}] train_total={train_total_loss:.6f} "
                        f"train_recon={train_recon_loss:.6f} train_anom={train_anom:.6f} train_div={train_div:.6f} "
                        f"val_loss={val_loss:.6f} val_anom={val_score:.6f} val_div={val_div:.6f} "
                        f"time={elapsed:.1f}s"
                    )
                    if checkpoint_manager is not None:
                        checkpoint_path = checkpoint_manager.save_epoch(
                            epoch=epoch,
                            model=base_model,
                            train_loss=train_total_loss,
                            val_loss=val_loss,
                            val_anom=val_score,
                            elapsed_time=elapsed,
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
                    if is_main_process:
                        print(
                            f"  ↳ new best model (val_anom={best_val_anom:.6f}, val_loss={best_val_loss:.6f})"
                        )

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

            print("\nEvaluating on test datasets...")

            test_scores = evaluate_tests_and_plot(
                model,
                test_loaders,
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
                print(
                    f"  {name:30s}: recon_loss={metrics['recon_loss']:.6f} "
                    f"anomaly_score={metrics['anomaly_score']:.6f}{div_str}{auc_str}{f1_str}"
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
