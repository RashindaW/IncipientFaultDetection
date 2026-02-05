#!/bin/bash
# =============================================================================
# FULLY AUTOMATED HP Search - All 120 experiments
# Uses bash background jobs with distributed GPU allocation
# =============================================================================

set -e

WORKDIR="/mnt/datassd3/rashinda/DySTGAT"
PYTHON="/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$WORKDIR/logs/hpsearch_$TIMESTAMP"

cd "$WORKDIR"
mkdir -p "$LOGDIR"

echo "==================================================================="
echo "DySTGAT HP Search - Fully Automated"
echo "==================================================================="
echo "Timestamp: $TIMESTAMP"
echo "Log directory: $LOGDIR"
echo ""
echo "GPU Allocation:"
echo "  GPU 0: TEP (40 exp, 10 parallel)"
echo "  GPU 1: ASHRAE (40 exp, 10 parallel)"
echo "  GPU 2: IMS-Raw exp 1-20 (5 parallel, shared)"
echo "  GPU 3: IMS-Raw exp 21-40 (5 parallel, shared)"
echo ""

# Function to run experiments with max parallelism for a dataset
run_dataset() {
    local DATASET=$1
    local GPU=$2
    local MAX_PARALLEL=$3
    local START_EXP=$4
    local END_EXP=$5
    local PIDS=()

    echo "[$DATASET GPU$GPU] Starting experiments $START_EXP-$END_EXP ($MAX_PARALLEL parallel)..."

    for exp in $(seq $START_EXP $END_EXP); do
        # Wait if we have MAX_PARALLEL jobs running
        while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
            NEW_PIDS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
            if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
                sleep 5
            fi
        done

        # Launch experiment in background
        $PYTHON run_hpsearch.py --dataset-key $DATASET --cuda-device $GPU --experiments $exp \
            > "$LOGDIR/${DATASET}_gpu${GPU}_exp${exp}.log" 2>&1 &
        PIDS+=($!)
        echo "[$DATASET GPU$GPU] Launched experiment $exp (PID: ${PIDS[-1]})"
    done

    # Wait for all remaining jobs
    echo "[$DATASET GPU$GPU] Waiting for remaining experiments to complete..."
    for pid in "${PIDS[@]}"; do
        wait $pid 2>/dev/null || true
    done
    echo "[$DATASET GPU$GPU] All experiments completed!"
}

# Launch all datasets in parallel
echo "[1/4] Starting TEP on GPU 0 (40 experiments, 10 parallel)..."
run_dataset tep 0 10 1 40 &
PID_TEP=$!

echo "[2/4] Starting ASHRAE on GPU 1 (40 experiments, 10 parallel)..."
run_dataset ashrae 1 10 1 40 &
PID_ASHRAE=$!

echo "[3/4] Starting IMS-Raw (1-20) on GPU 2 (20 experiments, 5 parallel)..."
run_dataset ims-raw 2 5 1 20 &
PID_IMSRAW_1=$!

echo "[4/4] Starting IMS-Raw (21-40) on GPU 3 (20 experiments, 5 parallel)..."
run_dataset ims-raw 3 5 21 40 &
PID_IMSRAW_2=$!

echo ""
echo "==================================================================="
echo "All HP search jobs launched!"
echo "==================================================================="
echo ""
echo "Process IDs:"
echo "  TEP (GPU 0):          $PID_TEP"
echo "  ASHRAE (GPU 1):       $PID_ASHRAE"
echo "  IMS-Raw 1-20 (GPU 2): $PID_IMSRAW_1"
echo "  IMS-Raw 21-40 (GPU 3): $PID_IMSRAW_2"
echo ""
echo "Monitor with:"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $LOGDIR/*.log"
echo ""

# Wait for all to complete
echo "Waiting for all experiments to complete..."
wait $PID_TEP $PID_ASHRAE $PID_IMSRAW_1 $PID_IMSRAW_2

echo ""
echo "==================================================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "==================================================================="
echo ""
echo "Results saved to:"
echo "  results/tep/hpsearch_*"
echo "  results/ashrae/hpsearch_*"
echo "  results/ims-raw/hpsearch_*"
echo ""
echo "Run aggregation with:"
echo "  $PYTHON run_hpsearch.py --dataset-key tep --aggregate-only --results-dir results/tep/hpsearch_*"
echo "  $PYTHON run_hpsearch.py --dataset-key ashrae --aggregate-only --results-dir results/ashrae/hpsearch_*"
echo "  $PYTHON run_hpsearch.py --dataset-key ims-raw --aggregate-only --results-dir results/ims-raw/hpsearch_*"
