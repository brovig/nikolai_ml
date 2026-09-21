"""Standalone ONNX export (CPU) for the trained Nikolai voice.

Mirrors piper_train/export_onnx.py with two torch-2.8 fixes:
  1. allowlist pathlib for weights_only checkpoint loading,
  2. run the export on CPU (checkpoint restores model_g on cuda, dummy
     inputs are CPU).
"""
import pathlib
import sys
from typing import Optional

import torch

torch.serialization.add_safe_globals([pathlib.Path, pathlib.PosixPath])

from piper_train.vits.lightning import VitsModel

OPSET_VERSION = 15

CKPT = "/nikolai_ml/checkpoints/lightning_logs/version_8/checkpoints/epoch=999-step=454000.ckpt"
OUT = "/nikolai_ml/onnx/nikolai.onnx"

torch.manual_seed(1234)
model = VitsModel.load_from_checkpoint(CKPT, dataset=None).to("cpu")
model_g = model.model_g

num_symbols = model_g.n_vocab
num_speakers = model_g.n_speakers
model_g.eval()

with torch.no_grad():
    model_g.dec.remove_weight_norm()

def infer_forward(text, text_lengths, scales, sid=None):
    audio = model_g.infer(
        text,
        text_lengths,
        noise_scale=scales[0],
        length_scale=scales[1],
        noise_scale_w=scales[2],
        sid=sid,
    )[0].unsqueeze(1)
    return audio

model_g.forward = infer_forward

dummy_input_length = 50
sequences = torch.randint(0, num_symbols, (1, dummy_input_length), dtype=torch.long)
sequence_lengths = torch.LongTensor([dummy_input_length])
sid: Optional[torch.LongTensor] = None
if num_speakers > 1:
    sid = torch.LongTensor([0])
scales = torch.FloatTensor([0.667, 1.0, 0.8])
dummy_input = (sequences, sequence_lengths, scales, sid)

torch.onnx.export(
    model=model_g,
    args=dummy_input,
    f=OUT,
    verbose=False,
    dynamo=False,  # legacy exporter (torch 2.8 default needs the onnx pkg)
    opset_version=OPSET_VERSION,
    input_names=["input", "input_lengths", "scales", "sid"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "phonemes"},
        "input_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 1: "time"},
    },
)
print("EXPORT_DONE ->", OUT)
