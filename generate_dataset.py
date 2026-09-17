#!/usr/bin/env python3
"""
Генерация датасета для Piper через Balabolka Console (Wine, SAPI4).
Голос: Acapela Elan TTS Nicolai
"""
import os
import re
import subprocess
import sys
import argparse
from pathlib import Path

# === НАСТРОЙКИ ===
BALCON_PATH = os.path.expanduser("~/.wine/drive_c/Program Files (x86)/Balabolka/balcon.exe")
VOICE_NAME = "Nicolai"
DICT_PATH = os.path.expanduser("~/.wine/drive_c/Program Files (x86)/Balabolka/michelangelo.dic")
INPUT_TEXT_FILE = "source_text.txt"
OUTPUT_DIR = "dataset_nikolai"

MIN_SENTENCE_LEN = 15
MAX_SENTENCE_LEN = 250

# === ИНИЦИАЛИЗАЦИЯ ===
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Генерация датасета для Piper")
    parser.add_argument(
        "--max-phrases", 
        type=int, 
        default=None,
        help="Максимальное количество фраз для генерации (по умолчанию: все)"
    )
    return parser.parse_args()

def check_dependencies():
    if subprocess.run(["which", "wine"], capture_output=True).returncode != 0:
        sys.exit("❌ Не найден wine. Установи: sudo pacman -S wine")
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        sys.exit("❌ Не найден ffmpeg. Установи: sudo pacman -S ffmpeg")
    if not os.path.exists(BALCON_PATH):
        sys.exit(f"❌ Не найден balcon.exe: {BALCON_PATH}")
    if not os.path.exists(DICT_PATH):
        print(f"⚠️  Словарь не найден: {DICT_PATH} (будем работать без него)")
    print("✅ Зависимости найдены.")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('…', '.')
    text = re.sub(r'https?://\S+', '', text)
    return text.strip()

def split_into_sentences(text):
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    
    for s in sentences:
        s = s.strip()
        if not s:
            continue
            
        if MIN_SENTENCE_LEN <= len(s) <= MAX_SENTENCE_LEN:
            result.append(s)
        elif len(s) > MAX_SENTENCE_LEN:
            parts = re.split(r'(?<=[,;:])\s+', s)
            chunk = ""
            for p in parts:
                if len(chunk) + len(p) + 1 < MAX_SENTENCE_LEN:
                    chunk = (chunk + " " + p).strip()
                else:
                    if chunk and len(chunk) >= MIN_SENTENCE_LEN:
                        result.append(chunk)
                    chunk = p
            if chunk and len(chunk) >= MIN_SENTENCE_LEN:
                result.append(chunk)
    
    return result

def generate_wav(text: str, wav_path: Path) -> bool:
    txt_path = wav_path.with_suffix(".tmp.txt")
    txt_path.write_text(text, encoding="utf-8")
    
    try:
        cmd = [
            "wine", BALCON_PATH,
            "-f", str(txt_path),
            "-w", str(wav_path),
            "-n", VOICE_NAME,
            "-fr", "22", "-bt", "16", "-ch", "1",
            "--encoding", "utf8"
        ]
        
        if os.path.exists(DICT_PATH):
            cmd.extend(["-d", DICT_PATH])
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60,
            env={**os.environ, "WINEDEBUG": "-all"}
        )
        
        if result.returncode != 0:
            return False
        
        if not wav_path.exists() or wav_path.stat().st_size < 1000:
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        return False
    finally:
        if txt_path.exists():
            txt_path.unlink()

def normalize_wav(wav_path: Path):
    """Нормализуем громкость через ffmpeg."""
    temp_path = wav_path.with_suffix(".norm.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-ar", "22050", "-ac", "1",
        "-c:a", "pcm_s16le",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(temp_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Проверяем, что ffmpeg создал файл
        if not temp_path.exists():
            print(f"\n⚠️  ffmpeg не создал файл, оставляем оригинал")
            return
        
        # Проверяем, что файл не пустой
        if temp_path.stat().st_size < 1000:
            print(f"\n⚠️  Нормализованный файл пустой, оставляем оригинал")
            temp_path.unlink()
            return
        
        # Заменяем оригинал на нормализованный
        temp_path.replace(wav_path)
        
    except Exception as e:
        print(f"\n⚠️  Error in normalize_wav: {e}")
        if temp_path.exists():
            temp_path.unlink()

def main():
    args = parse_args()
    max_phrases = args.max_phrases
    
    check_dependencies()
    
    if not os.path.exists(INPUT_TEXT_FILE):
        sys.exit(f"❌ Не найден {INPUT_TEXT_FILE}. Создай файл с текстом.")
    
    text = Path(INPUT_TEXT_FILE).read_text(encoding="utf-8")
    sentences = split_into_sentences(text)
    
    # Ограничиваем количество фраз, если указан параметр
    if max_phrases is not None:
        sentences = sentences[:max_phrases]
        print(f"🧪 ТЕСТОВЫЙ РЕЖИМ: генерируем только {max_phrases} фраз")
    
    if len(sentences) < 10:
        print(f"⚠️  Найдено всего {len(sentences)} фраз.")
        response = input("Продолжить? (y/n): ")
        if response.lower() != 'y':
            sys.exit("Отменено.")
    
    print(f"📝 Найдено {len(sentences)} фраз для озвучки.")
    print(f"📁 Датасет будет в: {OUTPUT_DIR}/")
    print("⏱️  Примерное время: 1–3 часа (для полного датасета)\n")
    
    success_count = 0
    for i, sentence in enumerate(sentences):
        wav_name = f"nikolai_{i:05d}.wav"
        lab_name = f"nikolai_{i:05d}.lab"
        wav_path = Path(OUTPUT_DIR) / wav_name
        lab_path = Path(OUTPUT_DIR) / lab_name
        
        # Пропускаем, если уже сделано
        if wav_path.exists() and lab_path.exists():
            success_count += 1
            continue
        
        # Прогресс-бар
        progress = (i + 1) / len(sentences) * 100
        print(f"\r[{progress:5.1f}%] [{i+1}/{len(sentences)}] {wav_name}", end="", flush=True)
        
        if generate_wav(sentence, wav_path):
            normalize_wav(wav_path)
            lab_path.write_text(sentence, encoding="utf-8")
            success_count += 1
        else:
            if wav_path.exists():
                wav_path.unlink()
    
    print(f"\n\n🎉 Готово!")
    print(f"✅ Успешно: {success_count}/{len(sentences)}")
    
    if success_count > 0:
        total_size = sum(f.stat().st_size for f in Path(OUTPUT_DIR).glob('*.wav'))
        print(f"📊 Размер датасета: {total_size / 1024 / 1024:.1f} MB")
    
    print(f"\n🚀 Следующий шаг: обучение Piper")

if __name__ == "__main__":
    main()