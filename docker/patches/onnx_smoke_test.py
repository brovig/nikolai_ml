"""Smoke test: synthesize a sentence with the exported ONNX via onnxruntime,
phonemizing exactly like piper_train/preprocess.py does."""
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
from piper_phonemize import phonemize_espeak, phoneme_ids_espeak

config = json.load(open('/nikolai_ml/preprocessed/config.json'))

text = 'Привет! Это голос Николая.'
all_phonemes = phonemize_espeak(text, 'ru')
phonemes = [p for sent in all_phonemes for p in sent]
print('phonemes:', phonemes)
phoneme_ids = phoneme_ids_espeak(phonemes)
print('ids:', phoneme_ids)

input_ids = np.array([phoneme_ids], dtype=np.int64)
lengths = np.array([len(phoneme_ids)], dtype=np.int64)
scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

sess = ort.InferenceSession('/nikolai_ml/onnx/nikolai.onnx', providers=['CPUExecutionProvider'])
outs = sess.run(None, {'input': input_ids, 'input_lengths': lengths, 'scales': scales})
audio = outs[0].flatten()
sr = config['audio']['sample_rate']
sf.write('/nikolai_ml/samples_final/onnx_test.wav', audio, sr)
print('wrote onnx_test.wav', f'{len(audio) / sr:.2f}s')
