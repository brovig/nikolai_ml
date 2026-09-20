FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=UTC \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

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
        setuptools==80.9.0 \
        wheel==0.45.1 \
    && python -m pip install \
        uv==0.5.24

# ------------------------------------------------------------
# PyTorch GPU stack
#
# Intentionally installed BEFORE Piper.
# Piper's requirements.txt contains torch<2, but we use the
# known working training stack and install Piper with --no-deps.
# ------------------------------------------------------------

RUN uv pip install \
        torch==2.5.1+cu124 \
        torchvision==0.20.1+cu124 \
        torchaudio==2.5.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# Piper direct dependencies
#
# These correspond to Piper commit:
# 73c04d81d5590ecc46e522de3601ce7fb29fc2be
#
# Piper itself is installed later with --no-deps so it cannot
# downgrade/replace the GPU stack above.
# ------------------------------------------------------------

RUN uv pip install \
        cython==0.29.37 \
        piper-phonemize==1.1.0 \
        librosa==0.10.2.post1 \
        numpy==1.26.4 \
        onnxruntime==1.20.1

# ------------------------------------------------------------
# Training/runtime compatibility dependencies
# ------------------------------------------------------------

RUN uv pip install \
        pytorch-lightning==1.9.0 \
        torchmetrics==0.11.4 \
        tensorboard==2.18.0 \
        scipy==1.11.4 \
        soundfile==0.12.1 \
        pillow==11.0.0

# ------------------------------------------------------------
# Fetch EXACT Piper source revision
# ------------------------------------------------------------

RUN git clone https://github.com/rhasspy/piper.git /tmp/piper \
    && cd /tmp/piper \
    && git checkout --detach 73c04d81d5590ecc46e522de3601ce7fb29fc2be \
    && git submodule update --init --recursive


# ------------------------------------------------------------
# Replace Piper lightning.py with our reproducible version
# ------------------------------------------------------------

COPY patches/lightning.py \
    /tmp/piper/src/python/piper_train/vits/lightning.py

    
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