# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Системные зависимости
RUN apt update && \
    apt install -y -qq --no-install-recommends \
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
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем uv глобально (в 22.04 это работает без танцев с бубном)
RUN python3 -m pip install uv

# Создаём виртуальное окружение
RUN python3 -m venv /app/.venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 2. PyTorch 2.5.1+cu124 (поддержка sm_61 для GTX 1060)
RUN uv pip install --no-deps \
        torch==2.5.1+cu124 \
        torchvision==0.20.1+cu124 \
        torchaudio==2.5.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

RUN uv pip install \
        sympy filelock jinja2 networkx fsspec mpmath typing-extensions

# 3. Базовые зависимости для обучения
RUN uv pip install \
        pytorch-lightning \
        librosa \
        soundfile \
        numpy==1.26.4 \
        scipy==1.11.4

# 4. Установка Piper
WORKDIR /tmp

RUN wget -qO piper.zip https://github.com/rhasspy/piper/archive/refs/heads/master.zip && \
    unzip -q piper.zip && \
    mv piper-master piper && \
    rm piper.zip

# В Ubuntu 22.04 (Python 3.10) готовое колесо piper-phonemize скачается и установится МГНОВЕННО!
RUN uv pip install piper-phonemize && \
    uv pip install -e ./piper/src/python && \
    rm -rf ./piper

# 5. Финальная настройка
WORKDIR /nikolai_ml

ENV PYTHONUNBUFFERED=1

ENTRYPOINT []
CMD ["tail", "-f", "/dev/null"]