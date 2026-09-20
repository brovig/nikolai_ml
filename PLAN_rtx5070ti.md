# Piper "Nikolai" TTS — Handoff Plan: re-target training to RTX 5070 Ti

> This document is the complete handoff for continuing the project on the **host
> machine with the RTX 5070 Ti 16 GB**. Read top-to-bottom before the first session.
> It is committed to the `v100-tuning` branch, so it travels with the git bundle.

---

## 1. What this project is

Train a **Piper VITS TTS voice** for the Russian single-speaker voice **"Nikolai"**.

- **Dataset**: 3 812 wav/lab pairs, 22 050 Hz, mono, 16-bit. Synthesized from the
  Windows SAPI4 voice *Acapela Elan TTS Nicolai* via Balabolka under Wine
  (`generate_dataset.py`), loudness-normalized with `ffmpeg loudnorm`.
- **Trainer**: official Piper at **pinned commit `73c04d81d5590ecc46e522de3601ce7fb29fc2be`**
  (VITS generator + multi-period discriminator GAN, PyTorch Lightning), with a
  **custom-patched `piper_train/vits/lightning.py`**
  (`docker/patches/lightning.py`) that adds:
  - `[VRAM]` logging (allocated/reserved/peak every 10 optimizer steps, peaks reset per epoch),
  - `[PERF]` CUDA-Event benchmark (samples/sec, batches/min, GPU vs wall time, every 50 batches),
  - `cudnn.benchmark=True`, `pin_memory`, `persistent_workers`, `prefetch_factor=2`,
  - a `--num-workers` CLI arg (added here, default 8).
- **Friend's baseline** (for reference only): GTX 1060 6 GB, shrunk model
  (96/96/384/4/2, Piper "quality x-low"), batch 6-8, ~**8 samples/sec**.

The quality tiers in Piper's `__main__.py`:

| `--quality` | effect |
|---|---|
| `x-low` | hidden/inter 96, filter 384 (friend's 1060 config) |
| `medium` (default) | defaults from `lightning.py`: **192/192/768/6/2** |
| `high` | resblock "1", upsample_initial_channel 512, bigger upsample kernels |

**Decision: use `medium`** (full default architecture) — full Piper voice quality
without overfitting risk on a 3.8k-sample dataset, and affordable on 16 GB VRAM.

---

## 2. Hardware situation & why we're moving

| Machine | GPU | Notes |
|---|---|---|
| This VM | Tesla V100 32 GB (sm_70) | Runs an LLM inference service (`ninfer-serve`, holds ~31 GB VRAM). Drivers incompatible with the host's 5070 Ti. |
| **Host (target)** | **RTX 5070 Ti 16 GB (Blackwell, sm_120)** | 2–3× faster expected. **Requires PyTorch ≥ 2.7 with cu128.** |

The current Docker image pins **torch 2.5.1+cu124 + PyTorch Lightning 1.9** —
that stack does **not** support Blackwell (sm_120). So the host run needs a
**modernized Dockerfile** (Section 5). The model code itself is plain PyTorch and
moves unchanged.

---

## 3. Repository state (what is already done)

- Upstream remote: `git@github.com:pelkoa-glitch/nikolai_ml.git` (friend's private repo).
- Branch to continue from: **`v100-tuning`** (head `0fdbb31`), contains:
  - `1892ca4` (friend): "fix cp patch command" — ⚠️ this commit actually **broke** the
    Dockerfile (`COPY docker/patches/lightning.py` does not resolve, build context is `./docker`).
  - `0fdbb31` (us, author `brovig <brovig.f@gmail.com>`): "V100 tuning":
    - Dockerfile: reverted to `COPY patches/lightning.py` ✅,
    - `docker/patches/lightning.py`: added `--num-workers` CLI arg (default 8).
      It auto-wires into `VitsModel`: Piper's `__main__.py` does
      `VitsModel(**vars(args))` and `VitsModel.__init__` already takes `num_workers`.
    - `run_preprocess.sh`, `run_train.sh` (see Sections 7-8),
    - `.gitignore`: added `dataset_nikolai.tar.gz`, `*.log`, `__pycache__/`.
- **Not committed** (gitignored, transferred separately):
  - `dataset_nikolai.tar.gz` (≈600 MB) — the dataset,
  - `PLAN_rtx5070ti.md` is committed with the final bundle build.

### Dataset verification (done on the VM)

- 3 812 `.wav` + 3 812 `.lab` + `metadata.csv(+.bak)` in `dataset_nikolai/`.
- All wavs: 22 050 Hz, mono, 16-bit, 2-8 s.
- All labs: single line, **no leading quotes/stray characters** (the bug the
  friend hit earlier in 33 files — already fixed upstream in this dataset).
- 767 MB extracted.

---

## 4. Transferring everything to the host

Two files must be copied from the VM to the host:

```
/home/llm/nikolai_ml/dataset_nikolai.tar.gz          # dataset (~600 MB)
/home/llm/nikolai_ml_nikolai_v100_tuning.bundle      # full git history + v100-tuning branch (~33 KB)
```

(e.g. `scp user@vm-host:/home/llm/... ./` from the host).

On the host:

```bash
# 1) Prereqs
docker --version
docker compose version
nvidia-smi            # driver must be new enough for Blackwell (>= ~570)
docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi   # GPU passthrough test

# 2) Restore the repo from the bundle (no access to the private remote needed)
mkdir nikolai_ml && cd nikolai_ml
git init
git fetch /path/to/nikolai_ml_nikolai_v100_tuning.bundle v100-tuning
git checkout v100-tuning
# set identity:
git config user.name "brovig"
git config user.email "brovig.f@gmail.com"

# 3) Dataset
cp /path/to/dataset_nikolai.tar.gz .
tar -xzf dataset_nikolai.tar.gz
# expect: dataset_nikolai/ with 3812 wav + 3812 lab
```

Host resources needed: ~25-30 GB free disk (image ~10 GB, preprocessed ~2-3 GB,
checkpoints), 16 GB+ RAM, 6+ CPU cores.

---

## 5. Modernized Dockerfile for the 5070 Ti

Work on a **new branch `rtx5070ti`** from `v100-tuning`. Keep the build context
`./docker` and `COPY patches/lightning.py` (NEVER `docker/patches/...` — that was
the friend's bug, section 10.1).

Rewrite `docker/Dockerfile` stages (keep the same structure/comments style):

1. **Base**: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
   (cu128 required for Blackwell wheels; ubuntu 24.04 is fine, has python3.12).
2. **System deps**: same list (python3-dev, venv, build-essential, git, wget,
   ca-certificates, ffmpeg, espeak-ng, libespeak-ng-dev, unzip).
3. **venv + uv** as before.
4. **PyTorch**: `uv pip install torch==2.8.* torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
   (verify exact latest 2.8.x with cu128; torch must report sm_120 support:
   `torch.cuda.get_arch_list()` must include `sm_120`).
5. **Piper direct deps** (keep pinned, they are version-agnostic):
   `piper-phonemize==1.1.0`, `librosa==0.10.2.post1`, `numpy` (let pip resolve a
   recent one; keep 1.26.x only if librosa complains), `onnxruntime` (latest stable),
   `cython` — start with a modern 3.x; the `monotonic_align` build is the one
   place that may fight us (see 8).
6. **Lightning 2.x**: `pytorch-lightning==2.2.*` (last 2.2 for maximum API
   stability) + `torchmetrics==1.*` + `tensorboard` + `scipy` + `soundfile` + `pillow`.
7. **Piper source**: same as before — clone, `git checkout --detach
   73c04d81d5590ecc46e522de3601ce7fb29fc2be`, submodules.
8. **Patch**: `COPY patches/lightning.py /tmp/piper/src/python/piper_train/vits/lightning.py`
9. **Install Piper**: `uv pip install --no-deps -e /tmp/piper/src/python`
   (MUST be `--no-deps`: old setup.py pins `torch<2`).
10. **monotonic_align**: same recipe as the V100 Dockerfile
    (`build_ext --inplace` + copy `core*.so` + `__init__.py`). If the old
    `setup.py` fails against torch 2.8 (C++/CUDA flag issues), minimal edits are
    acceptable — it's a tiny self-contained extension.
11. `WORKDIR /nikolai_ml`, `CMD ["tail","-f","/dev/null"]` unchanged.

`docker-compose.yaml`: keep as-is (context `./docker`, nvidia device passthrough,
shm 12 GB, volume `./:/nikolai_ml`). Optional: add
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to `environment`.

### Post-build smoke tests (in the container)

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_arch_list())          # must contain 'sm_120'
import pytorch_lightning as pl; print(pl.__version__)
import piper_train, piper_phonemize, librosa
from monotonic_align import minimum_distance_alignment  # import check
```

---

## 6. Porting the patched `lightning.py` to Lightning 2.x

Low risk, but verify this checklist (grep the file):

- ✅ `save_hyperparameters()` is already called in `__init__` → `self.hparams`
  works in 2.x unchanged.
- Hooks `on_train_batch_start` / `on_train_batch_end` / `on_train_epoch_start`:
  signatures unchanged in 2.x. ✅
- `training_step(batch, batch_idx, optimizer_idx)`, `validation_step`,
  `configure_optimizers`: unchanged. ✅
- Grep for removed 1.9-only APIs: `training_epoch_end`, `test_epoch_end`,
  `predict_epoch_end`, `self.trainer.progress_bar_*`, `weights_summary` — expect
  none; fix any hits.
- `Trainer.from_argparse_args` + `Trainer.add_argparse_args` (in Piper's
  `__main__.py`): fine in 2.x. Note the resume flag changed name
  (`--ckpt_path` → `--resume_from_checkpoint`).
- Keep the perf/VRAM logging code exactly as-is (pure torch + print).
- Compile-check after edits: `python -m py_compile docker/patches/lightning.py`.

First training run: **precision 32** (default). Only after the fp32 baseline is
stable, try `--precision 16` as a separate experiment (Blackwell FP16/TC is the
big upside, but GAN loss scaling can destabilize — compare loss curves &
samples/sec before committing).

---

## 7. Preprocessing (CPU-only, a few minutes)

Inside the container:

```bash
bash /nikolai_ml/run_preprocess.sh
# = python -m piper_train.preprocess \
#     --input-dir /nikolai_ml/dataset_nikolai \
#     --output-dir /nikolai_ml/preprocessed \
#     --language ru --sample-rate 22050 \
#     --dataset-format ljspeech --single-speaker --max-workers 8
```

Verify afterwards:

```bash
python - <<'EOF'
import json
c = json.load(open('/nikolai_ml/preprocessed/config.json'))
print(c['num_symbols'], c['num_speakers'], c['audio']['sample_rate'])
n = sum(1 for _ in open('/nikolai_ml/preprocessed/dataset.jsonl'))
print('records:', n)   # expect 3812
EOF
```

Expected: `num_speakers=1`, `sample_rate=22050`, 3812 records.
(Friend's earlier sanity numbers: max phoneme_ids 637, no concatenation artifacts.)

---

## 8. Training

### 8.1 Batch-size probe (16 GB card)

Start at **batch 16**; read the `[VRAM]` lines (`peak_reserved`); raise to 24 / 32
while `peak_reserved` stays **≤ ~13 GB** (leave headroom for allocator spikes and
the validation pass). Then compare `[PERF] samples/sec` between the candidates on
the same training segment (batches 50-200). **samples/sec is the metric — not
VRAM utilization** (batch 8 wasn't faster than 6 on the 1060 either).

### 8.2 Command

`run_train.sh` from the VM (batch 16, adjust after the probe):

```bash
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
# NO channel flags on purpose → medium architecture (192/192/768/6/2)
```

Run detached so it survives disconnects:

```bash
docker exec -d piper-train-nikolai bash -lc 'cd /nikolai_ml && nohup bash run_train.sh > train.log 2>&1 &'
tail -f /home/.../nikolai_ml/train.log
```

### 8.3 Throughput expectations & ETA

- 3 812 samples; 5% validation → 3 621 train / 190 val / 5 test (fixed for comparability).
- Epoch length: `3621 / batch_size` batches (remember: **2 optimizer steps per
  batch** — generator + discriminator — so `global_step ≈ 2 × batches`).
- V100 small-model estimate was ~8 samples/sec on 1060; medium model on 5070 Ti
  should land far higher. Compute ETA = `3812 × 1000 / (samples/sec from [PERF])`
  after the first `[PERF]` line and sanity-check it before the night run.

### 8.4 Disk / checkpoints

Checkpoints every 50 epochs → 20 files under `checkpoints/`. After the **first**
checkpoint (epoch 50), check its size; if 20 × size > ~15 GB, raise
`--checkpoint-epochs` to 100. Never delete checkpoints during a run (needed to
resume).

---

## 9. Monitoring

- `tail -f train.log` — watch:
  - `[PERF] epoch=… batch=… GPU=…s wall=…s samples/sec=… batches/min=…` (every 50 batches),
  - `[VRAM] epoch=… step=… stage=TRAIN allocated=… peak=…` (every 10 steps),
  - loss lines `loss_g_all`, `loss_disc_all`, val losses.
- Tensorboard: `tensorboard --logdir checkpoints/lightning_logs --host 0.0.0.0 --port 6006`
  (or on host: `python -m tensorboard.main --logdir ...`).
- GPU: `nvidia-smi` / `nvidia-smi -l 5`.
- Resume after a crash/stop: add `--resume_from_checkpoint checkpoints/<last>.ckpt`
  to the train command (Lightning 2.x flag name).

---

## 10. Known gotchas (friend's + ours) — read before debugging

1. **Docker COPY context**: build context is `./docker`, so the Dockerfile must
   say `COPY patches/lightning.py`, never `COPY docker/patches/lightning.py`
   (commit `1892ca4` did exactly this mistake; fixed in `0fdbb31`).
2. **Piper pins `torch<2` in its setup.py** → Piper must always be installed with
   `--no-deps`, on top of our pre-installed GPU stack.
3. **`global_step` ≠ number of dataset batches** — two optimizer steps per batch
   (G then D). Any per-step math must be divided by 2.
4. **Don't optimize for VRAM fill** — optimize for samples/sec; bigger batch only
   helps if samples/sec actually rises.
5. **Wall-clock timing is useless without CUDA sync** — use the CUDA-Event logger
   already in `lightning.py` (sync only every 50 batches).
6. **Blackwell**: if `torch.cuda.is_available()` is False or the build complains
   about `sm_120`, the torch wheel is wrong (need cu128 wheels, torch ≥ 2.7).
7. **`.lab` text bugs** (leading quotes / concatenation) silently break training
   metadata — dataset was verified clean here, but re-verify with the 2-liner in
   Section 7 if anything looks odd.
8. **espeak voice `ru`** must be available for phonemization (image ships
   espeak-ng; piper-phonemize bundles its own espeak data — don't "fix" either
   ad hoc).
9. **PyTorch allocator vs nvidia-smi**: `nvidia-smi` shows more than
   `torch.cuda.memory_reserved` because of CUDA context etc. Trust the `[VRAM]`
   logger for decisions.
10. **`expandable_segments:True`** stays on (reduces fragmentation OOMs on big batches).

---

## 11. First-session checklist on the host (for the agent)

1. `git log --oneline -3` → head must be `0fdbb31` on `v100-tuning`; identity
   `brovig <brovig.f@gmail.com>`; read this file fully.
2. Verify host prereqs (Section 4.1), incl. `--gpus all` passthrough test.
3. `git checkout -b rtx5070ti`; modernize `docker/Dockerfile` per Section 5.
4. `docker compose build` → run smoke tests (Section 5).
5. Port-check `lightning.py` (Section 6); `py_compile`.
6. `docker compose up -d`; run preprocessing (Section 7); verify outputs.
7. Batch-size probe (Section 8.1) — brief 30-60 s runs, read `[VRAM]`/`[PERF]`.
8. Launch full 1000-epoch run detached (Section 8.2); report: chosen batch,
   samples/sec, VRAM peak, ETA, first loss values.
9. Optionally schedule the `--precision 16` experiment after the fp32 baseline
   is confirmed stable (compare over the same batch range, not just startup).

---

## 12. Optional follow-ups (later)

- **FP16 / `--precision 16`** experiment (Blackwell TC upside; validate GAN stability).
- **`--quality high`** variant if medium output disappoints (heavier decoder, ~2× slower).
- **ONNX export** of the finished generator:
  `python -m piper_train.export_onnx` (Piper script, exists in the pinned commit)
  → usable with `piper` / `kokoro-onnx`-style runtimes.
- **Interoperability note**: if the V100 run was ever started here, its
  checkpoints (Lightning 1.9) should load into the 2.x trainer; if Lightning
  balks at the version metadata, load `state_dict` manually into a fresh model
  and continue.
- Long-term: friend intends a full Dockerfile rework — this modernized file is a
  good basis for that.

---

## 13. Key file map

| Path | Purpose |
|---|---|
| `docker/Dockerfile` | Image build (to modernize on host) |
| `docker/patches/lightning.py` | The custom `VitsModel` (VRAM + PERF logging) — patched into Piper at build time |
| `docker-compose.yaml` | Container: GPU passthrough, repo mounted at `/nikolai_ml` |
| `run_preprocess.sh` | Preprocess (CPU) |
| `run_train.sh` | Training entrypoint |
| `generate_dataset.py` | Dataset synthesis (Balabolka/Wine) — not needed for training |
| `prepare_filelist.py` | Legacy filelist helper (not used by the Docker flow) |
| `nikolai_config.json` | Legacy config file (not used by the CLI flow) |
| `chat_summary.md` | Friend's tuning history (1060 baseline, methodology) |
| `dataset_nikolai.tar.gz` | The dataset (transferred separately, gitignored) |
