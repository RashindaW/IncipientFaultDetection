#!/bin/bash
#
# Training script for PRONTO 15-Variable Merged Dataset
#
# This uses the new pronto_merged dataset adapter which loads from:
# data/pronto/pronto_benchmark/Pre-processed data/Process data/15var_merged/
#
# Dataset details:
# - 15 variables (4 conditioning + 11 measurement)
# - Combined flow rates (Air In, Water In instead of separate In1/In2)
# - Training: Normal + Air Blockage
# - Validation: Air Leakage
# - Test: Normal, Slugging, Air Blockage, Air Leakage, Diverted Flow
#

/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python train_dystgat.py \
  --dataset-key pronto_merged \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --weight-decay 1e-5 \
  --window-size 30 \
  --train-stride 1 \
  --val-stride 5 \
  --test-stride 1 \
  --use-spectral-view \
  --freq-embed-dim 24 \
  --freq-band-mix mlp \
  --freq-use-log \
  --freq-use-spectral-features \
  --fuse-mode gated \
  --divergence-type js \
  --anomaly-weight 0.5 \
  --lambda-div 0.1 \
  --task reconstruction \
  --use-amp \
  --cuda-device 2 \
  --checkpoint-dir runs/pronto_merged_spectral \
  --seed 42
