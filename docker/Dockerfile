# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# Системные зависимости (включая espeak-ng для Piper)
# ============================================================
RUN apt update && \
    apt install -y -qq --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        ffmpeg \
        espeak-ng \
        wget \
        ca-certificates \
        build-essential \
        git \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем uv
RUN python3 -m pip install --break-system-packages uv

# ============================================================
# PyTorch 2.5.1+cu124 (последняя с поддержкой sm_61)
# ============================================================
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-deps \
        torch==2.5.1+cu124 \
        torchvision==0.20.1+cu124 \
        torchaudio==2.5.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

# Зависимости PyTorch
RUN . /app/.venv/bin/activate && \
    uv pip install \
        sympy \
        filelock \
        jinja2 \
        networkx \
        fsspec \
        mpmath \
        typing-extensions

# ============================================================
# Зависимости для обучения Piper
# ============================================================
RUN . /app/.venv/bin/activate && \
    uv pip install \
        pytorch-lightning \
        librosa \
        soundfile \
        numpy==1.26.4 \
        scipy==1.11.4

# ============================================================
# Клонируем Piper, устанавливаем зависимости и сразу удаляем исходники
# ============================================================
WORKDIR /tmp

RUN git clone https://github.com/rhasspy/piper.git && \
    . /app/.venv/bin/activate && \
    # Используем стандартный pip для piper-phonemize, чтобы обойти строгие проверки manylinux в uv
    pip install piper-phonemize && \
    # Устанавливаем сам модуль обучения через uv
    uv pip install -e ./piper/src/python && \
    # УДАЛЯЕМ клонированный репозиторий для экономии места в образе!
    rm -rf ./piper

# ============================================================
# Возвращаем рабочую директорию
# ============================================================
WORKDIR /nikolai_ml

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT []
CMD ["tail", "-f", "/dev/null"]