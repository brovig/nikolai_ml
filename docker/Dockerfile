FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# NOTE (rtx5070ti): the plan suggested ubuntu24.04/python3.12, but
# piper-phonemize==1.1.0 ships no cp312 manylinux x86_64 wheel (cp310 is the
# newest for this target), and monotonic_align's setup.py relies on
# `distutils` which was removed in Python 3.12. Staying on ubuntu22.04
# (python3.10) keeps both buildable with zero source edits.

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=UTC \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=10

WORKDIR /app

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        build-essential \
        git \
        wget \
        ca-certificates \
        ffmpeg \
        espeak-ng \
        libespeak-ng-dev \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python environment
# ------------------------------------------------------------

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        pip==24.0 \
        setuptools==68.0.0 \
        wheel==0.45.1 \
    && python -m pip install \
        uv==0.5.24

# ------------------------------------------------------------
# PyTorch GPU stack (Blackwell / sm_120)
#
# Intentionally installed BEFORE Piper.
# Piper's requirements.txt contains torch<2, but we use a modern
# cu128 stack (required for RTX 5070 Ti / Blackwell) and install
# Piper with --no-deps.
# ------------------------------------------------------------

# UV_HTTP_TIMEOUT / UV_HTTP_RETRIES (set in ENV): the host network is
# flaky; uv's defaults (30 s / 5 retries) caused mid-download timeouts.
RUN uv pip install \
        torch==2.8.0+cu128 \
        torchvision==0.23.0+cu128 \
        torchaudio==2.8.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------
# Piper direct dependencies
#
# These correspond to Piper commit:
# 73c04d81d5590ecc46e522de3601ce7fb29fc2be
#
# piper-phonemize 1.1.0 has a cp310 manylinux wheel (hence python3.10 base).
# Cython is kept at 0.29.x: monotonic_align is built with it and it is the
# one spot in the whole image that may fight modern toolchains.
# ------------------------------------------------------------

RUN uv pip install \
        cython==0.29.37 \
        piper-phonemize==1.1.0 \
        librosa==0.10.2.post1 \
        numpy==1.26.4 \
        onnxruntime

# ------------------------------------------------------------
# Training/runtime compatibility dependencies
#
# Lightning 2.x (last 2.2 release for maximum API stability).
# ------------------------------------------------------------

RUN uv pip install \
        "pytorch-lightning==2.2.*" \
        "torchmetrics==1.*" \
        tensorboard \
        scipy \
        soundfile \
        pillow

# ------------------------------------------------------------
# Fetch EXACT Piper source revision
# ------------------------------------------------------------

RUN git clone https://github.com/rhasspy/piper.git /tmp/piper \
    && cd /tmp/piper \
    && git checkout --detach 73c04d81d5590ecc46e522de3601ce7fb29fc2be \
    && git submodule update --init --recursive


# ------------------------------------------------------------
# Replace Piper lightning.py with our reproducible version
#
# Build context is ./docker, so the path is patches/lightning.py.
# (NEVER docker/patches/... here — that was commit 1892ca4's bug.)
# ------------------------------------------------------------

COPY patches/lightning.py \
    /tmp/piper/src/python/piper_train/vits/lightning.py

# Piper's __main__.py calls Trainer.add_argparse_args / Trainer.from_argparse_args,
# both REMOVED in Lightning 2.x. Our patched version adds trainer args manually
# and constructs Trainer directly (resume passed to fit()).
COPY patches/__main__.py \
    /tmp/piper/src/python/piper_train/__main__.py

# Piper's __main__.py uses Trainer.add_argparse_args / Trainer.from_argparse_args,
# which were REMOVED in Lightning 2.x. Patch it to add trainer args manually
# and construct Trainer directly (see patches/__main__.py).
COPY patches/__main__.py \
    /tmp/piper/src/python/piper_train/__main__.py


# ------------------------------------------------------------
# Install Piper itself WITHOUT dependencies.
#
# This is intentional:
# setup.py for this Piper revision declares torch<2,
# which must NOT be allowed to replace our pinned PyTorch.
# ------------------------------------------------------------

RUN uv pip install --no-deps -e /tmp/piper/src/python

# ------------------------------------------------------------
# Build monotonic_align
# ------------------------------------------------------------

RUN cd /tmp/piper/src/python \
    && python piper_train/vits/monotonic_align/setup.py build_ext --inplace \
    && mkdir -p \
        /tmp/piper/src/python/piper_train/vits/monotonic_align/monotonic_align \
    && cp \
        piper_train/vits/monotonic_align/core*.so \
        piper_train/vits/monotonic_align/monotonic_align/ \
    && touch \
        piper_train/vits/monotonic_align/monotonic_align/__init__.py


WORKDIR /nikolai_ml

CMD ["tail", "-f", "/dev/null"]
