#!/bin/bash
# =============================================================================
# PRONTO Benchmark Experiments for DySTGAT vs DyEdgeGAT Comparison
# =============================================================================
#
# This script runs the complete evaluation suite needed to demonstrate
# DySTGAT outperforms DyEdgeGAT on the PRONTO benchmark.
#
# Usage:
#   chmod +x run_pronto_experiments.sh
#   ./run_pronto_experiments.sh
#
# Estimated time: ~2-4 hours (depending on GPU)
# =============================================================================

set -e  # Exit on error

# Configuration
GPU_ID=0
SEEDS=(42 123 456 789 1024)
EPOCHS=100
BATCH_SIZE=64
WINDOW_SIZE=15

RESULTS_DIR="experiments/pronto_comparison"
CHECKPOINTS_DIR="checkpoints/pronto_experiments"

mkdir -p $RESULTS_DIR
mkdir -p $CHECKPOINTS_DIR

echo "=============================================================================="
echo "PRONTO BENCHMARK EXPERIMENTS"
echo "=============================================================================="
echo "GPU: $GPU_ID"
echo "Seeds: ${SEEDS[@]}"
echo "Results: $RESULTS_DIR"
echo ""

# =============================================================================
# EXPERIMENT 1: DySTGAT (Full Model with Spectral View)
# =============================================================================
echo ""
echo "=============================================================================="
echo "EXPERIMENT 1: DySTGAT (Your Method - Full Model)"
echo "=============================================================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- DySTGAT Seed $seed ---"

    CUDA_VISIBLE_DEVICES=$GPU_ID python train_dystgat.py \
        --dataset-key pronto \
        --seed $seed \
        --window-size $WINDOW_SIZE \
        --train-stride 1 \
        --val-stride 5 \
        --test-stride 1 \
        --use-spectral-view \
        --freq-embed-dim 16 \
        --freq-band-mix mlp \
        --freq-use-log \
        --freq-use-spectral-features \
        --lambda-div 0.1 \
        --anomaly-weight 0.5 \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --use-amp \
        --save-model "$CHECKPOINTS_DIR/dystgat_seed${seed}.pt"

    # Evaluate
    python evaluate_pronto_comparison.py \
        --checkpoint "$CHECKPOINTS_DIR/dystgat_seed${seed}.pt" \
        --output-dir "$RESULTS_DIR/dystgat_seed${seed}" \
        --use-spectral-view \
        --freq-use-spectral-features \
        --use-amp
done

# =============================================================================
# EXPERIMENT 2: Baseline (No Spectral View = ~Original DyEdgeGAT)
# =============================================================================
echo ""
echo "=============================================================================="
echo "EXPERIMENT 2: Baseline (No Spectral View)"
echo "=============================================================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Baseline Seed $seed ---"

    CUDA_VISIBLE_DEVICES=$GPU_ID python train_dystgat.py \
        --dataset-key pronto \
        --seed $seed \
        --window-size $WINDOW_SIZE \
        --train-stride 1 \
        --val-stride 5 \
        --test-stride 1 \
        --anomaly-weight 0.5 \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --use-amp \
        --save-model "$CHECKPOINTS_DIR/baseline_seed${seed}.pt"

    # Evaluate
    python evaluate_pronto_comparison.py \
        --checkpoint "$CHECKPOINTS_DIR/baseline_seed${seed}.pt" \
        --output-dir "$RESULTS_DIR/baseline_seed${seed}" \
        --use-amp
done

# =============================================================================
# EXPERIMENT 3: Ablation Studies (Single seed for ablations)
# =============================================================================
echo ""
echo "=============================================================================="
echo "EXPERIMENT 3: Ablation Studies"
echo "=============================================================================="

ABLATION_SEED=42

# Ablation: No Divergence Loss
echo "--- Ablation: No Divergence Loss ---"
CUDA_VISIBLE_DEVICES=$GPU_ID python train_dystgat.py \
    --dataset-key pronto \
    --seed $ABLATION_SEED \
    --window-size $WINDOW_SIZE \
    --train-stride 1 \
    --val-stride 5 \
    --use-spectral-view \
    --freq-embed-dim 16 \
    --freq-band-mix mlp \
    --freq-use-log \
    --freq-use-spectral-features \
    --lambda-div 0.0 \
    --anomaly-weight 0.5 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --use-amp \
    --save-model "$CHECKPOINTS_DIR/ablation_no_div.pt"

python evaluate_pronto_comparison.py \
    --checkpoint "$CHECKPOINTS_DIR/ablation_no_div.pt" \
    --output-dir "$RESULTS_DIR/ablation_no_div" \
    --use-spectral-view \
    --freq-use-spectral-features \
    --use-amp

# Ablation: No Spectral Features
echo "--- Ablation: No Spectral Features ---"
CUDA_VISIBLE_DEVICES=$GPU_ID python train_dystgat.py \
    --dataset-key pronto \
    --seed $ABLATION_SEED \
    --window-size $WINDOW_SIZE \
    --train-stride 1 \
    --val-stride 5 \
    --use-spectral-view \
    --freq-embed-dim 16 \
    --freq-band-mix mlp \
    --freq-use-log \
    --lambda-div 0.1 \
    --anomaly-weight 0.5 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --use-amp \
    --save-model "$CHECKPOINTS_DIR/ablation_no_features.pt"

python evaluate_pronto_comparison.py \
    --checkpoint "$CHECKPOINTS_DIR/ablation_no_features.pt" \
    --output-dir "$RESULTS_DIR/ablation_no_features" \
    --use-spectral-view \
    --use-amp

# Ablation: Sum Fusion instead of Concat
echo "--- Ablation: Sum Fusion ---"
CUDA_VISIBLE_DEVICES=$GPU_ID python train_dystgat.py \
    --dataset-key pronto \
    --seed $ABLATION_SEED \
    --window-size $WINDOW_SIZE \
    --train-stride 1 \
    --val-stride 5 \
    --use-spectral-view \
    --freq-embed-dim 16 \
    --freq-band-mix mlp \
    --freq-use-log \
    --freq-use-spectral-features \
    --fuse-mode sum \
    --lambda-div 0.1 \
    --anomaly-weight 0.5 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --use-amp \
    --save-model "$CHECKPOINTS_DIR/ablation_sum_fusion.pt"

python evaluate_pronto_comparison.py \
    --checkpoint "$CHECKPOINTS_DIR/ablation_sum_fusion.pt" \
    --output-dir "$RESULTS_DIR/ablation_sum_fusion" \
    --use-spectral-view \
    --freq-use-spectral-features \
    --fuse-mode sum \
    --use-amp

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================
echo ""
echo "=============================================================================="
echo "AGGREGATING RESULTS"
echo "=============================================================================="

python -c "
import json
import numpy as np
from pathlib import Path
import pandas as pd

results_dir = Path('$RESULTS_DIR')

# Collect DySTGAT results
dystgat_results = []
for seed in [42, 123, 456, 789, 1024]:
    path = results_dir / f'dystgat_seed{seed}' / 'comparison_results.json'
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            metrics = data['per_fault']['faults_all']
            dystgat_results.append(metrics)

# Collect Baseline results
baseline_results = []
for seed in [42, 123, 456, 789, 1024]:
    path = results_dir / f'baseline_seed{seed}' / 'comparison_results.json'
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            metrics = data['per_fault']['faults_all']
            baseline_results.append(metrics)

# Compute statistics
def compute_stats(results, metric):
    values = [r[metric] for r in results]
    return np.mean(values), np.std(values)

print('=' * 70)
print('FINAL RESULTS SUMMARY')
print('=' * 70)
print()
print('DyEdgeGAT (from paper):')
print('  AUC: 0.83, F1: 0.64, F1*: 0.75')
print()

if dystgat_results:
    print(f'DySTGAT (yours) - {len(dystgat_results)} runs:')
    for metric in ['auc', 'f1', 'f1_star']:
        mean, std = compute_stats(dystgat_results, metric)
        print(f'  {metric}: {mean:.4f} +/- {std:.4f}')
    print()

if baseline_results:
    print(f'Baseline (no spectral) - {len(baseline_results)} runs:')
    for metric in ['auc', 'f1', 'f1_star']:
        mean, std = compute_stats(baseline_results, metric)
        print(f'  {metric}: {mean:.4f} +/- {std:.4f}')
    print()

# Statistical significance test
if len(dystgat_results) >= 2 and len(baseline_results) >= 2:
    print('Statistical Significance (paired t-test):')
    for metric in ['auc', 'f1', 'f1_star']:
        dystgat_vals = [r[metric] for r in dystgat_results]
        baseline_vals = [r[metric] for r in baseline_results]
        t_stat, p_val = stats.ttest_ind(dystgat_vals, baseline_vals)
        sig = 'YES' if p_val < 0.05 else 'NO'
        print(f'  {metric}: t={t_stat:.3f}, p={p_val:.4f} (significant: {sig})')
"

echo ""
echo "=============================================================================="
echo "EXPERIMENTS COMPLETE!"
echo "=============================================================================="
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Review per_fault_metrics.csv for detailed breakdown"
echo "2. Check ablation results to identify key contributions"
echo "3. Run on TEP dataset for additional validation"
