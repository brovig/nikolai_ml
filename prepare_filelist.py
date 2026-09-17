#!/usr/bin/env python3
"""Создаёт filelist.txt для обучения Piper."""
import os
from pathlib import Path

DATASET_DIR = "dataset_nikolai"
OUTPUT_FILE = "filelist.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for wav_file in sorted(Path(DATASET_DIR).glob("*.wav")):
        lab_file = wav_file.with_suffix(".lab")
        if lab_file.exists():
            text = lab_file.read_text(encoding="utf-8").strip()
            # Формат: путь_к_wav|текст
            out.write(f"{wav_file}|{text}\n")

count = sum(1 for _ in open(OUTPUT_FILE))
print(f"✅ Создан {OUTPUT_FILE} с {count} строками.")