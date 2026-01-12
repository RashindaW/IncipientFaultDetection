#!/usr/bin/env python3
"""
Dual-view analysis utilities for DySTGAT:
- Graph structure overlap/correlation between temporal and spectral views.
- Divergence score distributions (normal vs faults).
- Per-fault sensitivity comparing temporal-only vs dual-view checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from datasets import get_adapter
from dystgat.src.config import cfg
from train_dystgat import (
    forward_model,
    init_model,
    resolve_devices,
    unwrap_model,
    unpack_model_outputs,
    resolve_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-view analysis for DySTGAT.")
    parser.add_argument("--checkpoint", required=True, help="Dual-view model checkpoint.")
    parser.add_argument("--temporal-checkpoint", default=None, help="Temporal-only model checkpoint (optional).")
    parser.add_argument("--dataset-key", required=True, help="Dataset adapter key.")
    parser.add_argument("--data-dir", default=None, help="Override dataset directory.")
    parser.add_argument("--window-size", type=int, default=60, help="Sliding window size.")
    parser.add_argument("--train-stride", type=int, default=1, help="Stride for training windows.")
    parser.add_argument("--val-stride", type=int, default=1, help="Stride for validation windows.")
    parser.add_argument("--test-stride", type=int, default=1, help="Stride for test windows.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for analysis.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device.")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device index if using GPU.")
    parser.add_argument("--task", choices=["reconstruction", "prediction"], default="reconstruction")
    parser.add_argument("--pred-horizon", type=int, default=0, help="Prediction horizon when task=prediction.")
    parser.add_argument("--output-dir", default="analysis_outputs", help="Directory for analysis artifacts.")
    parser.add_argument("--max-batches", type=int, default=50, help="Max batches per split for analysis.")
    parser.add_argument("--skip-graph-analysis", action="store_true", help="Skip graph overlap analysis.")
    parser.add_argument("--skip-divergence-analysis", action="store_true", help="Skip divergence analysis.")
    parser.add_argument("--skip-fault-sensitivity", action="store_true", help="Skip fault sensitivity analysis.")

    parser.add_argument("--use-spectral-view", action="store_true", help="Enable spectral branch for dual-view model.")
    parser.add_argument("--freq-embed-dim", type=int, default=16, help="Spectral embedding dimension.")
    parser.add_argument("--freq-bins", type=int, default=0, help="Number of rFFT bins to keep (0 = all).")
    parser.add_argument(
        "--freq-band-mix",
        type=str,
        default="none",
        choices=["none", "conv", "mlp"],
        help="Band-mixing layer for spectral encoder.",
    )
    parser.add_argument("--freq-topk", type=int, default=None, help="Top-k neighbors for spectral graph.")
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
        help="Append spectral shape features to spectral embeddings.",
    )
    parser.add_argument("--share-gnn-weights", action="store_true", help="Share GNN weights across views.")
    parser.add_argument(
        "--fuse-mode",
        choices=["concat", "sum", "gated"],
        default="concat",
        help="Fusion strategy for temporal/spectral embeddings.",
    )
    parser.add_argument(
        "--divergence-type",
        choices=["js", "kl"],
        default="js",
        help="Divergence metric used for attention alignment.",
    )
    parser.add_argument(
        "--fault-keys",
        nargs="+",
        default=None,
        help="Optional fault keys (dataset-specific) for analysis.",
    )
    parser.add_argument(
        "--ashrae-feature-option",
        type=str,
        choices=["a", "b"],
        default=None,
        help="ASHRAE-only feature option.",
    )
    return parser.parse_args()


def build_model_args(args: argparse.Namespace, *, use_spectral_override: Optional[bool] = None) -> argparse.Namespace:
    model_args = argparse.Namespace(**vars(args))
    if use_spectral_override is not None:
        model_args.use_spectral_view = use_spectral_override
    return model_args


def load_model(
    args: argparse.Namespace,
    device: torch.device,
    adapter,
    checkpoint_path: str,
    *,
    use_spectral_override: Optional[bool] = None,
) -> torch.nn.Module:
    data_dir = args.data_dir or adapter.get_default_data_dir()
    if data_dir is None:
        raise ValueError("No data directory provided and adapter has no default.")
    feature_option = args.ashrae_feature_option
    control_vars = adapter.get_control_variables(data_dir, feature_option)
    n_nodes = adapter.measurement_count(feature_option)
    model_args = build_model_args(args, use_spectral_override=use_spectral_override)
    model = init_model(
        device,
        args.window_size,
        len(control_vars),
        n_nodes,
        args.task,
        args.pred_horizon,
        model_args=model_args,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    load_result = model.load_state_dict(checkpoint, strict=False)
    if load_result.missing_keys:
        print(f"[load] Missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[load] Unexpected keys: {load_result.unexpected_keys}")
    return model


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def collect_graph_stats(loader, model, device, max_batches: int):
    base_model = unwrap_model(model)
    sum_temp = None
    sum_freq = None
    overlaps = []
    correlations = []
    graph_count = 0

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_batches and idx >= max_batches:
                break
            outputs, batch_obj = forward_model(model, batch, device, return_graph=True)
            _, edge_index, edge_attr, aux = unpack_model_outputs(outputs)
            adj_freq = aux.get("adj_freq")
            attn_freq = aux.get("attn_freq")
            if adj_freq is None or attn_freq is None:
                continue

            b_graphs = int(batch_obj.batch.max().item()) + 1
            n_nodes = cfg.dataset.n_nodes

            dense_temp = base_model._dense_attn(edge_index, edge_attr.view(-1), b_graphs, n_nodes)
            dense_freq = base_model._dense_attn(adj_freq, attn_freq.view(-1), b_graphs, n_nodes)

            sum_temp = dense_temp.sum(dim=0) if sum_temp is None else sum_temp + dense_temp.sum(dim=0)
            sum_freq = dense_freq.sum(dim=0) if sum_freq is None else sum_freq + dense_freq.sum(dim=0)
            graph_count += dense_temp.size(0)

            temp_np = dense_temp.cpu().numpy()
            freq_np = dense_freq.cpu().numpy()
            for g in range(temp_np.shape[0]):
                temp_g = temp_np[g]
                freq_g = freq_np[g]
                mask_temp = temp_g > 0
                mask_freq = freq_g > 0
                union = mask_temp | mask_freq
                if union.sum() == 0:
                    overlaps.append(0.0)
                    correlations.append(0.0)
                    continue
                overlap = float((mask_temp & mask_freq).sum() / union.sum())
                overlaps.append(overlap)
                temp_vals = temp_g[union]
                freq_vals = freq_g[union]
                correlations.append(_safe_corrcoef(temp_vals, freq_vals))

    if sum_temp is None or graph_count == 0:
        return None

    mean_temp = (sum_temp / graph_count).cpu().numpy()
    mean_freq = (sum_freq / graph_count).cpu().numpy()
    stats = {
        "overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "overlap_std": float(np.std(overlaps)) if overlaps else 0.0,
        "corr_mean": float(np.mean(correlations)) if correlations else 0.0,
        "corr_std": float(np.std(correlations)) if correlations else 0.0,
    }
    return mean_temp, mean_freq, stats


def collect_divergence_scores(loader, model, device, max_batches: int) -> np.ndarray:
    scores = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_batches and idx >= max_batches:
                break
            outputs, _ = forward_model(model, batch, device, return_graph=True)
            _, _, _, aux = unpack_model_outputs(outputs)
            div = aux.get("divergence_score")
            if div is None:
                continue
            scores.extend(div.detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def collect_scores_and_labels(loader, model, device, task: str, max_batches: int) -> Tuple[np.ndarray, np.ndarray]:
    base_model = unwrap_model(model)
    scores = []
    labels = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_batches and idx >= max_batches:
                break
            outputs, batch_obj = forward_model(model, batch, device, return_graph=True)
            recon, edge_index, edge_attr, _ = unpack_model_outputs(outputs)
            target = resolve_target(batch_obj, recon, task)
            batch_scores = base_model.compute_anomaly_scores_per_sample(
                target, recon, edge_index, edge_attr
            )
            scores.append(batch_scores.cpu().numpy())
            labels.append(batch_obj.y.view(-1).cpu().numpy())
    if not scores:
        return np.array([]), np.array([])
    return np.concatenate(scores), np.concatenate(labels)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(scores.size) + 1
    pos_ranks = ranks[pos].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def per_fault_auc(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    fault_ids = sorted(set(labels[labels > 0].tolist()))
    results: Dict[str, float] = {}
    for fault_id in fault_ids:
        mask = (labels == 0) | (labels == fault_id)
        binary = (labels[mask] == fault_id).astype(np.int64)
        auc = roc_auc(scores[mask], binary)
        results[str(fault_id)] = auc
    return results


def main() -> None:
    args = parse_args()
    adapter = get_adapter(args.dataset_key)
    data_dir = args.data_dir or adapter.get_default_data_dir()
    if data_dir is None:
        raise ValueError("No data directory provided and adapter has no default.")

    device, _ = resolve_devices(args.device, args.cuda_device, None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_option = args.ashrae_feature_option
    train_loader, val_loader, test_loaders = adapter.create_dataloaders(
        window_size=args.window_size,
        batch_size=args.batch_size,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        test_stride=args.test_stride,
        data_dir=data_dir,
        num_workers=0,
        distributed=False,
        rank=0,
        world_size=1,
        baseline_from="val",
        severity_range=None,
        feature_option=feature_option,
        fault_keys=args.fault_keys,
        pred_horizon=args.pred_horizon,
    )

    baseline_loader = test_loaders.get("baseline", val_loader)
    fault_loader = None
    fault_key = None
    for key, loader in test_loaders.items():
        if key != "baseline":
            fault_loader = loader
            fault_key = key
            break

    dual_model = load_model(
        args,
        device,
        adapter,
        args.checkpoint,
        use_spectral_override=args.use_spectral_view,
    )

    if not args.skip_graph_analysis:
        if not unwrap_model(dual_model).use_spectral_view:
            print("[graph] Spectral view disabled; skipping graph overlap analysis.")
        else:
            baseline_stats = collect_graph_stats(baseline_loader, dual_model, device, args.max_batches)
            fault_stats = None
            if fault_loader is not None:
                fault_stats = collect_graph_stats(fault_loader, dual_model, device, args.max_batches)
            if baseline_stats is not None:
                mean_temp, mean_freq, stats = baseline_stats
                np.save(output_dir / "graph_mean_baseline_temp.npy", mean_temp)
                np.save(output_dir / "graph_mean_baseline_freq.npy", mean_freq)
                graph_stats = {"baseline": stats}
                if fault_stats is not None:
                    mean_temp_f, mean_freq_f, stats_f = fault_stats
                    np.save(output_dir / "graph_mean_fault_temp.npy", mean_temp_f)
                    np.save(output_dir / "graph_mean_fault_freq.npy", mean_freq_f)
                    graph_stats["faults"] = stats_f
                    graph_stats["faults_key"] = fault_key
                (output_dir / "graph_stats.json").write_text(
                    json.dumps(graph_stats, indent=2), encoding="utf-8"
                )

    if not args.skip_divergence_analysis:
        if not unwrap_model(dual_model).use_spectral_view:
            print("[divergence] Spectral view disabled; skipping divergence analysis.")
        else:
            baseline_scores = collect_divergence_scores(baseline_loader, dual_model, device, args.max_batches)
            if baseline_scores.size:
                np.save(output_dir / "divergence_scores_baseline.npy", baseline_scores)
            if fault_loader is not None:
                fault_scores = collect_divergence_scores(fault_loader, dual_model, device, args.max_batches)
                if fault_scores.size:
                    np.save(output_dir / "divergence_scores_fault.npy", fault_scores)

    if not args.skip_fault_sensitivity and args.temporal_checkpoint:
        temporal_model = load_model(
            args,
            device,
            adapter,
            args.temporal_checkpoint,
            use_spectral_override=False,
        )
        if fault_loader is None:
            print("[fault] No fault loader available; skipping fault sensitivity.")
        else:
            dual_scores, dual_labels = collect_scores_and_labels(
                fault_loader, dual_model, device, args.task, args.max_batches
            )
            temporal_scores, temporal_labels = collect_scores_and_labels(
                fault_loader, temporal_model, device, args.task, args.max_batches
            )
            if dual_scores.size == 0 or temporal_scores.size == 0:
                print("[fault] Empty scores; skipping fault sensitivity.")
                print(f"[fault] Dual scores: {dual_scores.size}, Temporal scores: {temporal_scores.size}")
                print(f"[fault] Dual labels: {dual_labels.size}, Temporal labels: {temporal_labels.size}")
                return
            dual_auc = per_fault_auc(dual_scores, dual_labels)
            temporal_auc = per_fault_auc(temporal_scores, temporal_labels)
            delta = {
                fault_id: dual_auc.get(fault_id, float("nan")) - temporal_auc.get(fault_id, float("nan"))
                for fault_id in sorted(set(dual_auc) | set(temporal_auc))
            }
            summary = {"dual": dual_auc, "temporal": temporal_auc, "delta": delta}
            (output_dir / "fault_auc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Analysis artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
