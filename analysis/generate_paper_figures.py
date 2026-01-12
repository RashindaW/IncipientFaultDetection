#!/usr/bin/env python3
"""
Generate paper figures from dual_view_analysis outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures for DySTGAT analysis.")
    parser.add_argument("--analysis-dir", required=True, help="Directory with analysis outputs.")
    parser.add_argument("--output-dir", default=None, help="Directory to save figures (defaults to analysis dir).")
    return parser.parse_args()


def _load_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


def plot_graph_comparison(temp: np.ndarray, freq: np.ndarray, output_path: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(temp, cmap="viridis")
    axes[0].set_title(f"{title_prefix} Temporal")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(freq, cmap="viridis")
    axes[1].set_title(f"{title_prefix} Spectral")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_divergence_distribution(baseline: np.ndarray, faults: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([baseline, faults], labels=["baseline", "faults"])
    ax.set_title("Divergence Score Distribution")
    ax.set_ylabel("Divergence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_fault_improvement(delta: dict, output_path: Path) -> None:
    fault_ids = sorted(delta, key=lambda k: float(k))
    values = [delta[fault_id] for fault_id in fault_ids]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(fault_ids, values)
    ax.set_title("Per-Fault AUC Improvement (Dual - Temporal)")
    ax.set_xlabel("Fault ID")
    ax.set_ylabel("AUC Delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_temp = _load_array(analysis_dir / "graph_mean_baseline_temp.npy")
    baseline_freq = _load_array(analysis_dir / "graph_mean_baseline_freq.npy")
    fault_temp = _load_array(analysis_dir / "graph_mean_fault_temp.npy")
    fault_freq = _load_array(analysis_dir / "graph_mean_fault_freq.npy")

    if baseline_temp is not None and baseline_freq is not None:
        plot_graph_comparison(
            baseline_temp,
            baseline_freq,
            output_dir / "fig_graph_comparison.png",
            "Baseline",
        )
    elif fault_temp is not None and fault_freq is not None:
        plot_graph_comparison(
            fault_temp,
            fault_freq,
            output_dir / "fig_graph_comparison.png",
            "Faults",
        )

    baseline_scores = _load_array(analysis_dir / "divergence_scores_baseline.npy")
    fault_scores = _load_array(analysis_dir / "divergence_scores_fault.npy")
    if baseline_scores is not None and fault_scores is not None:
        plot_divergence_distribution(
            baseline_scores,
            fault_scores,
            output_dir / "fig_divergence_distribution.png",
        )

    fault_auc_path = analysis_dir / "fault_auc.json"
    if fault_auc_path.exists():
        fault_auc = json.loads(fault_auc_path.read_text(encoding="utf-8"))
        delta = fault_auc.get("delta")
        if delta:
            plot_fault_improvement(delta, output_dir / "fig_fault_improvement.png")


if __name__ == "__main__":
    main()
