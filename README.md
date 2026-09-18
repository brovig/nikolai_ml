Start script:
```
cd /nikolai_ml

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m piper_train \
    --dataset-dir /nikolai_ml/preprocessed \
    --default_root_dir /nikolai_ml/checkpoints \
    --accelerator gpu \
    --devices 1 \
    --batch-size 8 \
    --validation-split 0.05 \
    --max_epochs 1000 \
    --checkpoint-epochs 50 \
    --hidden-channels 96 \
    --inter-channels 96 \
    --filter-channels 384 \
    --n-layers 4 \
    --n-heads 2
```

Preprocess script:
```
python -m piper_train.preprocess \
    --input-dir /nikolai_ml/dataset_nikolai \
    --output-dir /nikolai_ml/preprocessed \
    --language ru \
    --sample-rate 22050 \
    --dataset-format ljspeech \
    --single-speaker \
    --max-workers 4
```


docker exec -it piper-train-nikolai bash