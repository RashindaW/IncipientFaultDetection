#!/usr/bin/env bash
# Run all baseline methods on TEP dataset on GPU 0.
#
# All 6 methods run in parallel on GPU 0.
#
# Usage:
#   bash run_baselines_tep.sh

set -uo pipefail

PYTHON="/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python"
SCRIPT="train_baselines.py"
RUN_DIR="runs/tep_baselines_v1"

# Shared args (matching DySTGAT TEP config)
COMMON_ARGS=(
    --dataset-key tep
    --window-size 60
    --train-stride 1
    --val-stride 1
    --test-stride 1
    --seed 42
    --batch-size 64
    --epochs 200
    --early-stopping 20
    --num-workers 4
    --baseline-from test
    --learning-rate 1e-3
    --weight-decay 1e-5
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
echo "  Launching ${#METHODS[@]} baselines on GPU 0 (TEP dataset)"
echo "============================================================"

for method in "${METHODS[@]}"; do
    run_method "${method}" 0 &
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
