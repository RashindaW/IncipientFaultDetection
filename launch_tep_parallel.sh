#!/bin/bash
PYTHON=/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python
TRAIN=/mnt/datassd3/rashinda/DySTGAT/train_dystgat.py
LOGDIR=/mnt/datassd3/rashinda/DySTGAT/runs

echo "$(date): Launching TEP Phase 1 + Phase 1+2 on CUDA 0"

# TEP Phase 1
$PYTHON $TRAIN \
  --dataset-key tep --epochs 200 --batch-size 64 \
  --learning-rate 5e-5 --weight-decay 1e-5 --dropout 0.2 \
  --window-size 60 --train-stride 4 --val-stride 15 --test-stride 4 \
  --use-spectral-view --freq-embed-dim 24 --freq-band-mix mlp \
  --freq-use-log --freq-use-spectral-features --fuse-mode gated \
  --divergence-type js --anomaly-weight 0.5 --lambda-div 0.1 \
  --div-fusion-beta 0.3 --task reconstruction --use-amp --cuda-device 0 \
  --checkpoint-dir runs/tep_phase1 --seed 42 \
  --early-stopping --patience 20 --lr-scheduler cosine \
  --loss-type l2 --grad-clip-norm 0 --topology-mode neighbor_propagation \
  > "$LOGDIR/tep_phase1.log" 2>&1 &
echo "TEP Phase 1 PID: $!"

# TEP Phase 1+2
$PYTHON $TRAIN \
  --dataset-key tep --epochs 200 --batch-size 64 \
  --learning-rate 5e-5 --weight-decay 1e-5 --dropout 0.2 \
  --window-size 60 --train-stride 4 --val-stride 15 --test-stride 4 \
  --use-spectral-view --freq-embed-dim 24 --freq-band-mix mlp \
  --freq-use-log --freq-use-spectral-features --fuse-mode gated \
  --divergence-type js --anomaly-weight 0.5 --lambda-div 0.1 \
  --div-fusion-beta 0.3 --task reconstruction --use-amp --cuda-device 0 \
  --checkpoint-dir runs/tep_phase2_full --seed 42 \
  --early-stopping --patience 20 --lr-scheduler cosine \
  --loss-type l2 --grad-clip-norm 0 --topology-mode neighbor_propagation \
  --best-model-by val_anom --node-gru-input filtered \
  --gru-activation none --topology-error l2 \
  > "$LOGDIR/tep_phase2.log" 2>&1 &
echo "TEP Phase 1+2 PID: $!"

wait
echo "$(date): Both TEP jobs complete."
