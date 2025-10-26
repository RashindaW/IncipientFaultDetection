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
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as TorchDDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dyedgegat"))

from src.config import cfg
from src.data.dataset import RefrigerationDataset
from src.data.column_config import MEASUREMENT_VARS, CONTROL_VARS
from src.model.dyedgegat import DyEdgeGAT
from src.utils.checkpoint import EpochCheckpointManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast DyEdgeGAT training on subset of data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for DataLoaders")
    parser.add_argument("--train-stride", type=int, default=10, help="Stride for training windows")
    parser.add_argument("--val-stride", type=int, default=20, help="Stride for validation windows")
    parser.add_argument("--test-stride", type=int, default=20, help="Stride for test windows")
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Temporal window size (number of timesteps) for sliding windows.",
    )
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
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to store per-epoch checkpoints and metrics CSV.",
    )
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
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="Comma separated CUDA device indices for multi-GPU training (e.g., '0,1').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
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
        help="Distributed backend when launched with torchrun.",
    )
    return parser.parse_args()


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
                "Multi-GPU with --cuda-devices is no longer supported. Launch with torchrun for distributed training."
            )
        cuda_index = multi_device_ids[0]
        multi_device_ids = None

    if device_flag == "cpu":
        if cuda_index is not None:
            raise ValueError("CUDA device index specified but device='cpu'.")
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


def init_model(device: torch.device, window_size: int) -> DyEdgeGAT:
    cfg.set_dataset_params(
        n_nodes=len(MEASUREMENT_VARS),  # 142 nodes (measurement sensors)
        window_size=window_size,
        ocvar_dim=len(CONTROL_VARS),  # 10 control variables
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
    for module in model.modules():
        if isinstance(module, torch.nn.GRU):
            module.flatten_parameters = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    return model.to(device)


def build_dataloaders(
    args: argparse.Namespace,
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
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

    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


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


def compute_recon_loss(model: torch.nn.Module, batch, criterion, device) -> Tuple[torch.Tensor, Batch]:
    recon, batch_obj = forward_model(model, batch, device, return_graph=False)
    target = batch_obj.x.unsqueeze(-1)
    loss = criterion(recon, target)
    return loss, batch_obj


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    *,
    distributed: bool = False,
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for raw_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp_enabled):
            loss, batch_obj = compute_recon_loss(model, raw_batch, criterion, device)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = batch_obj.num_graphs
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    totals = torch.tensor([total_loss, total_samples], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    loss_sum, sample_sum = totals.tolist()
    return float(loss_sum / max(sample_sum, 1.0))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion,
    device,
    *,
    distributed: bool = False,
    amp_enabled: bool = False,
) -> Tuple[float, float]:
    model.eval()
    base_model = unwrap_model(model)
    total_loss = 0.0
    total_score = 0.0
    total_samples = 0

    for raw_batch in loader:
        with autocast("cuda", enabled=amp_enabled):
            (recon, edge_index, edge_attr), batch_obj = forward_model(
                model, raw_batch, device, return_graph=True
            )
            target = batch_obj.x.unsqueeze(-1)
            loss = criterion(recon, target)

        score = base_model.compute_topology_aware_anomaly_score(
            target, recon, edge_index, edge_attr
        )

        batch_size = batch_obj.num_graphs
        total_loss += loss.detach().item() * batch_size
        total_score += score.item() * batch_size
        total_samples += batch_size

    totals = torch.tensor([total_loss, total_score, total_samples], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    loss_sum, score_sum, sample_sum = totals.tolist()
    denom = max(sample_sum, 1.0)
    return float(loss_sum / denom), float(score_sum / denom)


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed_mode(args.dist_backend)
    is_main_process = (not distributed) or rank == 0

    try:
        if distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device, _ = resolve_devices(args.device, args.cuda_device, args.cuda_devices)

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        if is_main_process:
            print("=" * 80)
            print("FAST DYEDGEGAT TRAINING")
            print("=" * 80)
            print(f"Train files: {args.train_files}")
            print(f"Val file  : {args.val_file}")
            print(f"Test file : {args.test_file}")
            if distributed:
                print(f"Distributed across {world_size} GPUs")
            else:
                print(f"Device    : {device}")
            print("=" * 80)

        model = init_model(device, args.window_size)
        if distributed:
            model = TorchDDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )

        base_model = unwrap_model(model)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        scaler = GradScaler("cuda") if args.use_amp and device.type == "cuda" else None
        amp_enabled = scaler is not None

        train_loader, val_loader, test_loader = build_dataloaders(
            args,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            num_workers=args.num_workers,
        )

        best_val_loss = float("inf")
        best_state = None
        checkpoint_manager = None

        if is_main_process and args.checkpoint_dir:
            checkpoint_manager = EpochCheckpointManager(args.checkpoint_dir, prefix="dyedgegat")
            print(f"\nSaving per-epoch checkpoints to: {checkpoint_manager.run_path}")

        for epoch in range(1, args.epochs + 1):
            if distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            start = time.time()
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                distributed=distributed,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            val_loss, val_score = evaluate(
                model,
                val_loader,
                criterion,
                device,
                distributed=distributed,
                amp_enabled=amp_enabled,
            )
            elapsed = time.time() - start

            if is_main_process:
                print(
                    f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} val_anom={val_score:.6f} time={elapsed:.1f}s"
                )
                if checkpoint_manager is not None:
                    checkpoint_path = checkpoint_manager.save_epoch(
                        epoch=epoch,
                        model=base_model,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_anom=val_score,
                        elapsed_time=elapsed,
                    )
                    print(f"  ↳ checkpoint saved: {checkpoint_path.name}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

        if best_state is not None:
            base_model.load_state_dict(best_state)
            if is_main_process:
                print(f"\nBest validation loss: {best_val_loss:.6f}")
        elif is_main_process:
            print("\nWarning: No improvement over initial epoch.")

        test_loss, test_score = evaluate(
            model,
            test_loader,
            criterion,
            device,
            distributed=distributed,
            amp_enabled=amp_enabled,
        )
        if is_main_process:
            print(
                f"\nTest results on {args.test_file}: "
                f"recon_loss={test_loss:.6f} anomaly_score={test_score:.6f}"
            )
            print("\nDone.")
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
