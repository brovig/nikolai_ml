#!/usr/bin/env bash
# Run inside container: bash /nikolai_ml/run_preprocess.sh
set -euo pipefail
cd /nikolai_ml

python -m piper_train.preprocess \
    --input-dir /nikolai_ml/dataset_nikolai \
    --output-dir /nikolai_ml/preprocessed \
    --language ru \
    --sample-rate 22050 \
    --dataset-format ljspeech \
    --single-speaker \
    --max-workers 8
