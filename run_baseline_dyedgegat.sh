#!/bin/bash

# Run baseline DyEdgeGAT WITHOUT spectral view
# This should be closer to the original paper's architecture

echo "=============================================================================="
echo "Running BASELINE DyEdgeGAT (without spectral view)"
echo "This matches the original paper's architecture more closely"
echo "=============================================================================="

CUDA_VISIBLE_DEVICES=3 python train_dyedgegat.py \
    --dataset-key pronto \
    --window-size 15 \
    --anomaly-weight 0.5 \
    --epochs 30 \
    --batch-size 64 \
    --use-amp

# Note: Removed flags:
# --use-spectral-view (no dual-view)
# --freq-embed-dim 16 (not needed without spectral view)
# --freq-band-mix mlp (not needed without spectral view)
# --lambda-div 0.1 (no divergence loss without dual-view)

echo ""
echo "=============================================================================="
echo "Expected differences:"
echo "- Simpler architecture (no spectral branch)"
echo "- Fewer parameters (~50% less)"
echo "- Potentially lower scores (closer to paper's 0.80 AUC)"
echo "=============================================================================="


