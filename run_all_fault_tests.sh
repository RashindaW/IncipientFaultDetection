#!/bin/bash
# Script to test all 6 fault datasets and generate anomaly score plots
# This helps visualize incipient faults across different failure modes

set -e  # Exit on error

CHECKPOINT="checkpoints/dyedgegat_stride10.pt"
OUTPUT_DIR="outputs/anomaly_scores"
STRIDE=1
BATCH_SIZE=64
NUM_WORKERS=4

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Array of all fault datasets
FAULTS=(
    "Fault1_DisplayCaseDoorOpen"
    "Fault2_IceAccumulation"
    "Fault3_EvapValveFailure"
    "Fault4_MTEvapFanFailure"
    "Fault5_CondAPBlock"
    "Fault6_MTEvapAPBlock"
)

echo "=========================================="
echo "Testing Model on All Fault Datasets"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of Faults: ${#FAULTS[@]}"
echo "=========================================="
echo ""

# Process each fault dataset
for i in "${!FAULTS[@]}"; do
    FAULT="${FAULTS[$i]}"
    NUM=$((i+1))
    
    echo "[$NUM/${#FAULTS[@]}] Processing: $FAULT"
    echo "----------------------------------------"
    
    python plot_anomaly_scores.py \
        --checkpoint "$CHECKPOINT" \
        --dataset "$FAULT" \
        --stride "$STRIDE" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --output-dir "$OUTPUT_DIR" \
        --line-plot \
        --use-amp
    
    echo ""
    echo "✓ Completed: $FAULT"
    echo ""
done

echo "=========================================="
echo "All fault tests completed!"
echo "=========================================="
echo ""
echo "Generated files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR" | tail -n +2
echo ""
echo "Files generated per fault:"
echo "  - <fault>_anomaly_scores.csv      : Per-timestep scores"
echo "  - <fault>_timestamp_mean.csv      : Timestamp-averaged scores"
echo "  - <fault>_heatmap.png             : Heatmap visualization"
echo "  - <fault>_line.png                : Line plot (temporal progression)"

