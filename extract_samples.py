"""Extract the LAST audio clips from a TF event file by reading only the tail.

TFRecord format per record:
  [8B length][4B len_crc][N bytes payload][4B payload_crc]
Records are appended sequentially, so the last record ends exactly at EOF.
We read the tail, locate record boundaries by walking BACKWARDS from EOF
(a candidate start s is valid when uint64(buf[s:s+8]) == next_boundary - s - 16,
an 8-byte match => no false positives), then decode and keep the last audio
seen per tag.
"""
import glob
import os
import struct
import numpy as np
import soundfile as sf
try:
    from tensorboard.core import event_pb2
except ModuleNotFoundError:
    from tensorboard.compat.proto import event_pb2

TAIL = 32 * 1024 * 1024          # 32 MB tail — final epoch writes 5 clips (<400 KB each)
MAX_RECORD = 8 * 1024 * 1024     # sanity cap for a single record
WANT_TAGS = 5                    # we expect 5 test sentences

files = sorted(glob.glob('/nikolai_ml/checkpoints/lightning_logs/version_8/events.out.tfevents.*'))
assert files, 'no event file found'
path = files[-1]
size = os.path.getsize(path)

with open(path, 'rb') as f:
    f.seek(max(0, size - TAIL))
    buf = f.read()
end = len(buf)


def prev_boundary(q):
    """Given a boundary at offset q, find the start of the record that ends at q."""
    for s in range(q - 16, max(0, q - MAX_RECORD - 16), -1):
        (L,) = struct.unpack_from('<Q', buf, s)
        if L == q - s - 16 and L < MAX_RECORD:
            return s
    return None


# Walk backwards collecting record payloads until we have all 5 audio tags
# or we run out of the tail window.
last_audio = {}
q = end
n = 0
while q > 16:
    s = prev_boundary(q)
    if s is None:
        break
    payload = buf[s + 12:s + 12 + (q - s - 16)]
    e = event_pb2.Event()
    e.ParseFromString(payload)
    if e.HasField('summary'):
        for v in e.summary.value:
            if v.HasField('audio'):
                last_audio[v.tag] = (e.step, v.audio.encoded_audio_string, v.audio.sample_rate)
    n += 1
    q = s
    if len(last_audio) >= WANT_TAGS:
        break

print(f'records scanned (backwards): {n}, audio tags found: {len(last_audio)}', flush=True)
for i, (tag, (step, data, sr)) in enumerate(sorted(last_audio.items())):
    audio = np.frombuffer(data, dtype=np.int16)
    sr = int(sr) if sr else 22050
    out = f'/nikolai_ml/samples_final/{i}.wav'
    sf.write(out, audio, sr or 22050)
    print(f'wrote {out}  step={step}  {len(audio)/sr:.2f}s  tag={tag!r}', flush=True)
print('DONE', flush=True)
