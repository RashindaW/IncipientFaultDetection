#!/bin/bash
# Launch all 9 baselines on ASHRAE dataset, ALL simultaneously across 3 GPUs.
#
# Config matched to DySTGAT best ASHRAE run:
#   window=180, batch=64, epochs=200, LR=1e-3, WD=1e-5, patience=30
#   stride: train=4, val/test=8
#   Seed: 42
#
# Distribution (all 9 run simultaneously, 3 per GPU):
#   GPU 1: grelen, mtad_gat, usad
#   GPU 2: dyedgegat, gdn, lstm
#   GPU 3: ae, fnn, lstm_ae
#
# Usage:
#   bash launch_ashrae_baselines.sh              # run all 9 simultaneously
#   bash launch_ashrae_baselines.sh 0 ae         # run only AE on GPU 0

set -e

CHECKPOINT_DIR="runs/ashrae_baselines"
mkdir -p "$CHECKPOINT_DIR"

COMMON_ARGS="--dataset-key ashrae --window-size 180 --batch-size 64 --epochs 200 \
--learning-rate 1e-3 --weight-decay 1e-5 --early-stopping 30 --seed 42 \
--train-stride 4 --val-stride 8 --test-stride 8 \
--checkpoint-dir $CHECKPOINT_DIR"

run_one() {
    local gpu=$1
    local method=$2
    local extra_args="${3:-}"
    echo "[GPU $gpu] Starting: $method"
    python train_baselines.py --method "$method" --cuda-device "$gpu" $COMMON_ARGS $extra_args \
        > "${CHECKPOINT_DIR}/${method}.log" 2>&1
    echo "[GPU $gpu] Finished: $method"
}

# Single-method mode
if [ -n "$2" ]; then
    run_one "$1" "$2"
    exit 0
fi

# Launch all 9 simultaneously
echo "Launching all 9 ASHRAE baselines simultaneously across GPUs 1, 2, 3..."
echo "Logs: ${CHECKPOINT_DIR}/<method>.log"
echo ""

PIDS=()

# GPU 1
for m in grelen mtad_gat usad; do
    run_one 1 "$m" &
    PIDS+=($!)
done

# GPU 2
run_one 2 dyedgegat "--batch-size 32" &
PIDS+=($!)
for m in gdn lstm; do
    run_one 2 "$m" &
    PIDS+=($!)
done

# GPU 3
for m in ae fnn lstm_ae; do
    run_one 3 "$m" &
    PIDS+=($!)
done

echo "All 9 launched. PIDs: ${PIDS[*]}"
echo "Waiting for all to finish..."

wait "${PIDS[@]}"
echo ""
echo "All 9 ASHRAE baselines complete. Results in $CHECKPOINT_DIR/"
