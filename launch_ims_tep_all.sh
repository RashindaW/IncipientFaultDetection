#!/bin/bash
# Run IMS Phase 1, IMS Phase 1+2, TEP Phase 1, TEP Phase 1+2 sequentially on CUDA 0
PYTHON=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
TRAIN=/mnt/datassd3/rashinda/DySTGAT/train_dystgat.py
LOG=/mnt/datassd3/rashinda/DySTGAT/runs/ims_tep_sequential.log

echo "$(date): Starting sequential IMS + TEP runs on CUDA 0" >> "$LOG"

# 1. IMS Phase 1
echo "$(date): === IMS Phase 1 ===" >> "$LOG"
$PYTHON $TRAIN \
  --dataset-key ims-raw --window-size 1024 --train-stride 1 --val-stride 1 --test-stride 4 \
  --use-spectral-view --freq-embed-dim 32 --freq-band-mix mlp \
  --freq-use-log --freq-use-spectral-features --fuse-mode gated \
  --divergence-type js --lambda-div 0.1 --anomaly-weight 0.5 \
  --div-fusion-beta 0.3 --epochs 200 --batch-size 64 \
  --learning-rate 1e-4 --weight-decay 1e-5 --dropout 0.2 \
  --task reconstruction --use-amp --cuda-device 0 \
  --checkpoint-dir runs/ims_raw_phase1 --seed 42 \
  --early-stopping --patience 30 \
  --loss-type l2 --grad-clip-norm 0 --topology-mode neighbor_propagation \
  >> "$LOG" 2>&1
echo "$(date): IMS Phase 1 finished (exit $?)" >> "$LOG"

# 2. IMS Phase 1+2
echo "$(date): === IMS Phase 1+2 ===" >> "$LOG"
$PYTHON $TRAIN \
  --dataset-key ims-raw --window-size 1024 --train-stride 1 --val-stride 1 --test-stride 4 \
  --use-spectral-view --freq-embed-dim 32 --freq-band-mix mlp \
  --freq-use-log --freq-use-spectral-features --fuse-mode gated \
  --divergence-type js --lambda-div 0.1 --anomaly-weight 0.5 \
  --div-fusion-beta 0.3 --epochs 200 --batch-size 64 \
  --learning-rate 1e-4 --weight-decay 1e-5 --dropout 0.2 \
  --task reconstruction --use-amp --cuda-device 0 \
  --checkpoint-dir runs/ims_raw_phase2_full --seed 42 \
  --early-stopping --patience 30 \
  --loss-type l2 --grad-clip-norm 0 --topology-mode neighbor_propagation \
  --best-model-by val_anom --node-gru-input filtered \
  --gru-activation none --topology-error l2 \
  >> "$LOG" 2>&1
echo "$(date): IMS Phase 1+2 finished (exit $?)" >> "$LOG"

# 3. TEP Phase 1
echo "$(date): === TEP Phase 1 ===" >> "$LOG"
$PYTHON $TRAIN \
  --dataset-key tep --epochs 200 --batch-size 64 \
  --learning-rate 5e-5 --weight-decay 1e-5 --dropout 0.2 \
  --window-size 60 --train-stride 4 --val-stride 4 --test-stride 4 \
  --use-spectral-view --freq-embed-dim 24 --freq-band-mix mlp \
  --freq-use-log --freq-use-spectral-features --fuse-mode gated \
  --divergence-type js --anomaly-weight 0.5 --lambda-div 0.1 \
  --div-fusion-beta 0.3 --task reconstruction --use-amp --cuda-device 0 \
  --checkpoint-dir runs/tep_phase1 --seed 42 \
  --early-stopping --patience 20 --lr-scheduler cosine \
  --loss-type l2 --grad-clip-norm 0 --topology-mode neighbor_propagation \
  >> "$LOG" 2>&1
echo "$(date): TEP Phase 1 finished (exit $?)" >> "$LOG"

# 4. TEP Phase 1+2
echo "$(date): === TEP Phase 1+2 ===" >> "$LOG"
$PYTHON $TRAIN \
  --dataset-key tep --epochs 200 --batch-size 64 \
  --learning-rate 5e-5 --weight-decay 1e-5 --dropout 0.2 \
  --window-size 60 --train-stride 4 --val-stride 4 --test-stride 4 \
  --use-spectral-view --freq-embed-dim 24 --freq-band-mix mlp \
  --freq-use-log --freq-use-spectral-features --fuse-mode gated \
  --divergence-type js --anomaly-weight 0.5 --lambda-div 0.1 \
  --div-fusion-beta 0.3 --task reconstruction --use-amp --cuda-device 0 \
  --checkpoint-dir runs/tep_phase2_full --seed 42 \
  --early-stopping --patience 20 --lr-scheduler cosine \
  --loss-type l2 --grad-clip-norm 0 --topology-mode neighbor_propagation \
  --best-model-by val_anom --node-gru-input filtered \
  --gru-activation none --topology-error l2 \
  >> "$LOG" 2>&1
echo "$(date): TEP Phase 1+2 finished (exit $?)" >> "$LOG"

echo "$(date): All 4 runs complete." >> "$LOG"
