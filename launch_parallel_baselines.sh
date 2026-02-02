#!/bin/bash
# Launch parallel baseline training across 4 GPUs
# Each GPU handles one dataset for optimal parallelization
#
# IMPORTANT: This script only stops OUR baseline tmux session.
# Omar's processes (oawadall) will NOT be affected.

set -e

RESULTS_DIR="results/baselines/20260126_071141"
CONDA_ENV="rashindaNew-torch-env"
CONDA_INIT="source /opt/anaconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"

echo "=============================================="
echo "PARALLEL BASELINE TRAINING LAUNCHER"
echo "=============================================="
echo "Results directory: ${RESULTS_DIR}"
echo "Conda environment: ${CONDA_ENV}"
echo ""

# Step 1: Verify Omar's processes are running (sanity check)
echo "Step 1: Verifying Omar's processes are still running..."
omar_count=$(ps aux | grep oawadall | grep python | grep -v grep | wc -l)
echo "  Found ${omar_count} Python processes owned by oawadall"
if [ "$omar_count" -gt 0 ]; then
    echo "  Omar's processes are running (will NOT be affected)"
else
    echo "  Warning: No oawadall Python processes found"
fi
echo ""

# Step 2: Stop ONLY our current baseline tmux session
echo "Step 2: Stopping our existing baseline tmux session (if any)..."
if tmux has-session -t baselines 2>/dev/null; then
    echo "  Found 'baselines' session, sending Ctrl+C..."
    tmux send-keys -t baselines C-c
    sleep 3
    echo "  Killing 'baselines' session..."
    tmux kill-session -t baselines
    echo "  Session stopped."
else
    echo "  No 'baselines' session found (already stopped or never started)"
fi
echo ""

# Step 3: Re-verify Omar's processes
echo "Step 3: Verifying Omar's processes are STILL running..."
omar_count_after=$(ps aux | grep oawadall | grep python | grep -v grep | wc -l)
echo "  Found ${omar_count_after} Python processes owned by oawadall"
if [ "$omar_count_after" -eq "$omar_count" ]; then
    echo "  Omar's processes unaffected (same count: ${omar_count})"
else
    echo "  WARNING: Process count changed! Before: ${omar_count}, After: ${omar_count_after}"
fi
echo ""

# Step 4: Launch parallel tmux sessions
echo "Step 4: Launching 4 parallel tmux sessions..."
echo ""

# GPU 0: ims-raw (may have existing results, use --skip-existing)
echo "  GPU 0: ims-raw"
tmux new-session -d -s baselines_gpu0 "${CONDA_INIT} && cd /mnt/datassd3/rashinda/DySTGAT && python run_baselines_parallel.py --cuda-device 0 --datasets ims-raw --skip-existing --output-dir ${RESULTS_DIR} 2>&1 | tee ${RESULTS_DIR}/gpu0_log.txt"

# GPU 1: tep
echo "  GPU 1: tep"
tmux new-session -d -s baselines_gpu1 "${CONDA_INIT} && cd /mnt/datassd3/rashinda/DySTGAT && python run_baselines_parallel.py --cuda-device 1 --datasets tep --output-dir ${RESULTS_DIR} 2>&1 | tee ${RESULTS_DIR}/gpu1_log.txt"

# GPU 2: ashrae
echo "  GPU 2: ashrae"
tmux new-session -d -s baselines_gpu2 "${CONDA_INIT} && cd /mnt/datassd3/rashinda/DySTGAT && python run_baselines_parallel.py --cuda-device 2 --datasets ashrae --output-dir ${RESULTS_DIR} 2>&1 | tee ${RESULTS_DIR}/gpu2_log.txt"

# GPU 3: pronto
echo "  GPU 3: pronto"
tmux new-session -d -s baselines_gpu3 "${CONDA_INIT} && cd /mnt/datassd3/rashinda/DySTGAT && python run_baselines_parallel.py --cuda-device 3 --datasets pronto --output-dir ${RESULTS_DIR} 2>&1 | tee ${RESULTS_DIR}/gpu3_log.txt"

echo ""
sleep 2

# Step 5: Verify all sessions are running
echo "Step 5: Verifying all tmux sessions..."
echo ""
tmux list-sessions | grep baselines_gpu || echo "  Warning: No baselines_gpu sessions found!"
echo ""

# Step 6: Print monitoring commands
echo "=============================================="
echo "MONITORING COMMANDS"
echo "=============================================="
echo ""
echo "# Check all sessions:"
echo "tmux list-sessions | grep baselines_gpu"
echo ""
echo "# Attach to a session (e.g., GPU 0):"
echo "tmux attach -t baselines_gpu0"
echo ""
echo "# Watch GPU utilization:"
echo "watch -n 10 nvidia-smi"
echo ""
echo "# Count completed experiments:"
echo "watch -n 60 'find ${RESULTS_DIR} -name \"*_best.pt\" | wc -l'"
echo ""
echo "# View logs:"
echo "tail -f ${RESULTS_DIR}/gpu0_log.txt"
echo "tail -f ${RESULTS_DIR}/gpu1_log.txt"
echo "tail -f ${RESULTS_DIR}/gpu2_log.txt"
echo "tail -f ${RESULTS_DIR}/gpu3_log.txt"
echo ""
echo "# After completion, generate comparison:"
echo "python compare_baselines.py --results-dir ${RESULTS_DIR} --format all"
echo ""
echo "=============================================="
echo "LAUNCH COMPLETE"
echo "=============================================="
