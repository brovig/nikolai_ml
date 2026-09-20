#!/usr/bin/env bash
# Run inside container: bash /nikolai_ml/run_train.sh
# RTX 5070 Ti 16 GB: medium Piper architecture (192/192/768/6/2).
# ~4 GB VRAM is used by the desktop (X server), so keep [VRAM]
# peak_reserved <= ~11 GB after the batch-size probe.
set -euo pipefail
cd /nikolai_ml

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m piper_train \
    --dataset-dir /nikolai_ml/preprocessed \
    --default_root_dir /nikolai_ml/checkpoints \
    --accelerator gpu \
    --devices 1 \
    --batch-size 16 \
    --validation-split 0.05 \
    --num-workers 8 \
    --max_epochs 1000 \
    --checkpoint-epochs 50
