#!/bin/bash
# =============================================================================
# PRONTO 15-Variable Hyperparameter Search: DySTGAT vs DyEdgeGAT
# =============================================================================
#
# GPU Allocation (all GPUs run in parallel):
#   GPU 0: 6 parallel jobs (free GPU, most experiments)
#   GPU 1: 2 parallel jobs (S3DIS running)
#   GPU 2: 2 parallel jobs (S3DIS running)
#   GPU 3: 2 parallel jobs (S3DIS running)
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python"

LOG_DIR="runs/pronto_merged_spectral/hpsearch_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/hpsearch_master_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

# Base configuration
BASE_ARGS="--dataset-key pronto_merged \
    --epochs 200 \
    --batch-size 64 \
    --window-size 15 \
    --train-stride 1 \
    --val-stride 1 \
    --test-stride 1 \
    --split-mode segment_shuffle \
    --train-segments 0,1,3,5,9,7,8 \
    --val-segments 2,4 \
    --test-segments 6 \
    --divergence-type js \
    --task reconstruction \
    --use-amp \
    --checkpoint-dir runs/pronto_merged_spectral \
    --seed 42 \
    --early-stopping \
    --patience 20"

SPECTRAL_ARGS="--use-spectral-view --freq-embed-dim 24 --freq-band-mix mlp --freq-use-log --freq-use-spectral-features"
BASE_HP="--learning-rate 3e-5 --weight-decay 1e-5 --anomaly-weight 0.5 --fuse-mode gated"

# Run single experiment
run_exp() {
    local gpu=$1
    local name=$2
    shift 2
    local extra_args="$*"
    local log_file="$LOG_DIR/${name}_${TIMESTAMP}.log"

    log "[GPU $gpu] Starting: $name"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON train_dystgat.py $BASE_ARGS $extra_args > "$log_file" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        log "[GPU $gpu] Completed: $name"
    else
        log "[GPU $gpu] FAILED: $name (exit $status)"
    fi
}

# Run a queue of experiments on a specific GPU with parallelism limit
run_gpu_queue() {
    local gpu=$1
    local max_parallel=$2
    shift 2
    local experiments=("$@")

    local running=0
    local pids=()

    for exp in "${experiments[@]}"; do
        # Parse: "name|arg1|arg2|..."
        IFS='|' read -ra parts <<< "$exp"
        local name="${parts[0]}"
        local args="${parts[*]:1}"
        args="${args//|/ }"

        # Wait if at max parallel
        while [ $running -ge $max_parallel ]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}" 2>/dev/null || true
                    unset 'pids[i]'
                    ((running--)) || true
                fi
            done
            pids=("${pids[@]}")
            [ $running -ge $max_parallel ] && sleep 5
        done

        run_exp $gpu "$name" $args &
        pids+=($!)
        ((running++))
    done

    # Wait for remaining
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

log "========================================"
log "PRONTO HP Search Started"
log "Timestamp: $TIMESTAMP"
log "========================================"

# =============================================================================
# Define experiments for each GPU
# =============================================================================

# GPU 0: Baseline + Lambda-div + Anomaly-wt (partial) + Ablations = ~18 experiments
GPU0_EXPS=(
    # Phase 1: Baselines
    "baseline_dyedgegat|$BASE_HP|--lambda-div|0"
    "baseline_dystgat|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.2"
    # Phase 2A: Lambda-div
    "lambda_div_0.0|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.0"
    "lambda_div_0.05|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.05"
    "lambda_div_0.1|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.1"
    "lambda_div_0.2|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.2"
    "lambda_div_0.3|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.3"
    # Phase 2A: Anomaly weights (partial)
    "anomaly_wt_0.1|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|0.1"
    "anomaly_wt_0.3|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|0.3"
    "anomaly_wt_0.5|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|0.5"
    # Phase 2C: Ablations
    "ablation_no_spectral|$BASE_HP|--lambda-div|0"
    "ablation_no_divergence|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0"
    "ablation_no_spectral_features|--use-spectral-view|--freq-embed-dim|24|--freq-band-mix|mlp|--freq-use-log|$BASE_HP|--lambda-div|0.2"
    "ablation_concat_fusion|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|concat|--lambda-div|0.2|--anomaly-weight|0.5"
    "ablation_sum_fusion|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|sum|--lambda-div|0.2|--anomaly-weight|0.5"
)

# GPU 1: Anomaly weights (rest) + Fuse modes = 5 experiments
GPU1_EXPS=(
    "anomaly_wt_1.0|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|1.0"
    "anomaly_wt_2.0|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|2.0"
    "fuse_mode_concat|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|concat|--lambda-div|0.2|--anomaly-weight|0.5"
    "fuse_mode_sum|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|sum|--lambda-div|0.2|--anomaly-weight|0.5"
    "fuse_mode_gated|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|0.5"
)

# GPU 2: Learning rates + Window sizes = 8 experiments
GPU2_EXPS=(
    "lr_1e-5|$SPECTRAL_ARGS|--learning-rate|1e-5|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|0.5"
    "lr_3e-5|$SPECTRAL_ARGS|--learning-rate|3e-5|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|0.5"
    "lr_1e-4|$SPECTRAL_ARGS|--learning-rate|1e-4|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|0.5"
    "lr_3e-4|$SPECTRAL_ARGS|--learning-rate|3e-4|--weight-decay|1e-5|--fuse-mode|gated|--lambda-div|0.2|--anomaly-weight|0.5"
    "window_10|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.2|--window-size|10"
    "window_15|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.2|--window-size|15"
    "window_30|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.2|--window-size|30"
    "window_45|$SPECTRAL_ARGS|$BASE_HP|--lambda-div|0.2|--window-size|45"
)

# GPU 3: Freq embed dims + Band mix = 7 experiments
GPU3_EXPS=(
    "freq_embed_16|--use-spectral-view|--freq-embed-dim|16|--freq-band-mix|mlp|--freq-use-log|--freq-use-spectral-features|$BASE_HP|--lambda-div|0.2"
    "freq_embed_24|--use-spectral-view|--freq-embed-dim|24|--freq-band-mix|mlp|--freq-use-log|--freq-use-spectral-features|$BASE_HP|--lambda-div|0.2"
    "freq_embed_32|--use-spectral-view|--freq-embed-dim|32|--freq-band-mix|mlp|--freq-use-log|--freq-use-spectral-features|$BASE_HP|--lambda-div|0.2"
    "freq_embed_48|--use-spectral-view|--freq-embed-dim|48|--freq-band-mix|mlp|--freq-use-log|--freq-use-spectral-features|$BASE_HP|--lambda-div|0.2"
    "band_mix_none|--use-spectral-view|--freq-embed-dim|24|--freq-band-mix|none|--freq-use-log|--freq-use-spectral-features|$BASE_HP|--lambda-div|0.2"
    "band_mix_conv|--use-spectral-view|--freq-embed-dim|24|--freq-band-mix|conv|--freq-use-log|--freq-use-spectral-features|$BASE_HP|--lambda-div|0.2"
    "band_mix_mlp|--use-spectral-view|--freq-embed-dim|24|--freq-band-mix|mlp|--freq-use-log|--freq-use-spectral-features|$BASE_HP|--lambda-div|0.2"
)

log "GPU 0: ${#GPU0_EXPS[@]} experiments (6 parallel)"
log "GPU 1: ${#GPU1_EXPS[@]} experiments (2 parallel)"
log "GPU 2: ${#GPU2_EXPS[@]} experiments (2 parallel)"
log "GPU 3: ${#GPU3_EXPS[@]} experiments (2 parallel)"
log "Total: $((${#GPU0_EXPS[@]} + ${#GPU1_EXPS[@]} + ${#GPU2_EXPS[@]} + ${#GPU3_EXPS[@]})) experiments"
log "========================================"

# =============================================================================
# Launch all GPU queues in parallel
# =============================================================================

run_gpu_queue 0 6 "${GPU0_EXPS[@]}" &
PID0=$!

run_gpu_queue 1 2 "${GPU1_EXPS[@]}" &
PID1=$!

run_gpu_queue 2 2 "${GPU2_EXPS[@]}" &
PID2=$!

run_gpu_queue 3 2 "${GPU3_EXPS[@]}" &
PID3=$!

log "All GPU queues launched"
log "  GPU 0 PID: $PID0"
log "  GPU 1 PID: $PID1"
log "  GPU 2 PID: $PID2"
log "  GPU 3 PID: $PID3"

# Wait for all to complete
wait $PID0 && log "GPU 0 queue complete" || log "GPU 0 queue had errors"
wait $PID1 && log "GPU 1 queue complete" || log "GPU 1 queue had errors"
wait $PID2 && log "GPU 2 queue complete" || log "GPU 2 queue had errors"
wait $PID3 && log "GPU 3 queue complete" || log "GPU 3 queue had errors"

log ""
log "========================================"
log "PRONTO HP Search Completed!"
log "========================================"

COMPLETED=$(find runs/pronto_merged_spectral -name "detailed_test_metrics.csv" -newer "$MASTER_LOG" 2>/dev/null | wc -l)
log "Experiments with test metrics: $COMPLETED"
log "Results: runs/pronto_merged_spectral/"
log "Master log: $MASTER_LOG"
log ""
log "Next: python aggregate_pronto_results.py --output results_summary.csv"
