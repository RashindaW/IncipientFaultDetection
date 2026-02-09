#!/usr/bin/env bash
# Run all baseline methods on PRONTO in parallel across 4 GPUs.
#
# GPU assignment (6 methods, 4 GPUs):
#   GPU 0: lstm_vae, mtad_gat, dyedgegat
#   GPU 1: usad
#   GPU 2: omnianomaly
#   GPU 3: gdn
#
# Segment split (same as DySTGAT v8):
#   train: 2,3,4,6,7,8,9   val: 0,1   test: 5
#
# Usage:
#   bash run_baselines_pronto.sh

set -uo pipefail

PYTHON="/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python"
SCRIPT="train_baselines.py"
RUN_DIR="runs/pronto_baselines_v8"

# Shared data/split args (same as DySTGAT v8)
COMMON_ARGS=(
    --dataset-key pronto
    --window-size 30
    --train-stride 1
    --val-stride 1
    --test-stride 1
    --split-mode segment_shuffle
    --train-segments 2,3,4,6,7,8,9
    --val-segments 0,1
    --test-segments 5
    --seed 42
    --batch-size 64
    --epochs 200
    --early-stopping 20
    --num-workers 4
    --baseline-from test
    --learning-rate 1e-3
    --weight-decay 1e-5
)

# Method → GPU mapping
declare -A GPU_MAP=(
    [lstm_vae]=0
    [usad]=1
    [omnianomaly]=2
    [gdn]=3
    [mtad_gat]=0
    [dyedgegat]=0
)

METHODS=(lstm_vae usad omnianomaly gdn mtad_gat dyedgegat)
PIDS=()
FAILED=()

run_method() {
    local method=$1
    local gpu=$2
    local ckpt_dir="${RUN_DIR}/${method}"
    mkdir -p "${ckpt_dir}"

    echo "[$(date +%H:%M:%S)] Starting ${method} on GPU ${gpu}"

    $PYTHON $SCRIPT \
        --method "${method}" \
        --cuda-device "${gpu}" \
        "${COMMON_ARGS[@]}" \
        --checkpoint-dir "${ckpt_dir}" \
        --save-model "${ckpt_dir}/best.pt" \
        > "${ckpt_dir}/train.log" 2>&1

    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] DONE  ${method} (GPU ${gpu})"
    else
        echo "[$(date +%H:%M:%S)] FAIL  ${method} (GPU ${gpu}, exit ${rc})"
    fi
    return $rc
}

echo "============================================================"
echo "  Launching ${#METHODS[@]} baselines across 4 GPUs"
echo "============================================================"

for method in "${METHODS[@]}"; do
    run_method "${method}" "${GPU_MAP[$method]}" &
    PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}"
echo ""

# Wait for all and collect failures
for i in "${!METHODS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        FAILED+=("${METHODS[$i]}")
    fi
done

echo ""
echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "  All ${#METHODS[@]} baselines finished successfully."
else
    echo "  ${#FAILED[@]}/${#METHODS[@]} baselines FAILED: ${FAILED[*]}"
    echo "  Check logs: ${RUN_DIR}/<method>/train.log"
    exit 1
fi
echo "============================================================"
