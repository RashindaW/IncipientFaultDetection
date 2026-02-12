#!/bin/bash
# Monitor GPU memory and launch ASHRAE Phase 1+2 when 25GB+ is free on any GPU (0-3)
LOG="/mnt/datassd3/rashinda/DySTGAT/runs/ashrae_phase2_monitor.log"
echo "$(date): Waiting for a GPU with 25GB+ free memory..." >> "$LOG"

while true; do
    for GPU_ID in 0 1 2 3; do
        FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | tr -d ' ')
        if [ -n "$FREE_MB" ] && [ "$FREE_MB" -ge 25000 ]; then
            echo "$(date): GPU $GPU_ID has ${FREE_MB}MB free. Launching ASHRAE Phase 1+2." >> "$LOG"
            /home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python /mnt/datassd3/rashinda/DySTGAT/train_dystgat.py \
              --dataset-key ashrae --ashrae-feature-option a --epochs 300 --batch-size 128 \
              --learning-rate 3e-4 --weight-decay 1e-5 --dropout 0.2 \
              --window-size 180 --train-stride 4 --val-stride 8 --test-stride 8 \
              --use-spectral-view --freq-embed-dim 24 --freq-band-mix mlp \
              --freq-use-log --freq-use-spectral-features --fuse-mode gated \
              --divergence-type js --anomaly-weight 0.5 --lambda-div 0.1 \
              --div-fusion-beta 0.3 --task reconstruction --use-amp --cuda-device "$GPU_ID" \
              --checkpoint-dir runs/ashrae_phase2_full --seed 42 \
              --early-stopping --patience 20 \
              --loss-type l2 --grad-clip-norm 0 --topology-mode neighbor_propagation \
              --best-model-by val_anom --node-gru-input filtered \
              --gru-activation none --topology-error l2 \
              >> "$LOG" 2>&1
            echo "$(date): ASHRAE Phase 1+2 finished (exit code $?)." >> "$LOG"
            exit 0
        fi
    done
    sleep 60
done
