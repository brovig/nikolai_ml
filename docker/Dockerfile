# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        ffmpeg \
        espeak-ng \
        libespeak-ng-dev \
        wget \
        ca-certificates \
        build-essential \
        git \
        unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir uv

RUN python3 -m venv /app/.venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# ============================================================
# PyTorch
# ============================================================

RUN uv pip install --no-deps \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Зависимости torch
RUN uv pip install \
    sympy \
    filelock \
    jinja2 \
    networkx \
    fsspec \
    mpmath \
    typing-extensions \
    six

# ============================================================
# Piper dependencies
# ============================================================

RUN uv pip install \
    "pytorch-lightning==1.9.0" \
    "torchmetrics==0.11.4" \
    "librosa>=0.9.2,<1" \
    "soundfile" \
    "numpy==1.26.4" \
    "scipy==1.11.4" \
    "six" \
    "pillow" \
    "Cython<3" \
    "onnxruntime"

# ============================================================
# Piper
# ============================================================

WORKDIR /tmp

RUN wget -qO piper.zip \
    https://github.com/rhasspy/piper/archive/refs/heads/master.zip && \
    unzip -q piper.zip && \
    mv piper-master piper && \
    rm piper.zip

RUN uv pip install piper-phonemize

# Устанавливаем Piper, но НЕ позволяем ему менять наш torch stack
RUN uv pip install --no-deps -e /tmp/piper/src/python

WORKDIR /tmp/piper/src/python/piper_train/vits/monotonic_align

RUN python setup.py build_ext --inplace && \
    mkdir -p monotonic_align && \
    cp piper_train/vits/monotonic_align/core*.so monotonic_align/

WORKDIR /nikolai_ml



WORKDIR /nikolai_ml

ENTRYPOINT []
CMD ["tail", "-f", "/dev/null"]