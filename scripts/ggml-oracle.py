# Writes GGML block samples with gguf-py's own dequantization (llama.cpp's Python package), the oracle for
# GgufBlocks: records of int32 type id, rows, cols, raw byte count; the raw blocks; rows*cols float32, little-endian.
#
#   python3 -m venv /tmp/gguf && /tmp/gguf/bin/pip install gguf numpy
#   /tmp/gguf/bin/python scripts/ggml-oracle.py \
#       dynamis-gpu-primitives/src/test/resources/org/dynamisengine/gpu/primitives/ggml-blocks.bin   # synthetic, committed
#   /tmp/gguf/bin/python scripts/ggml-oracle.py /tmp/synthetic.bin /tmp/real.bin model1.gguf ...         # real slices
#
# then run GgufBlocksTest and the Quantized conformance tests with -Dgguf.samples=/tmp/real.bin. The real slices are
# model weights: keep them out of the repository.
import sys, struct, numpy as np, gguf
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize
SIZES = {T.Q4_0: (32, 18), T.Q4_1: (32, 20), T.Q8_0: (32, 34), T.Q4_K: (256, 144), T.Q5_K: (256, 176), T.Q6_K: (256, 210)}

def record(out, qt, raw, rows, cols):
    vals = dequantize(np.frombuffer(raw, dtype=np.uint8), qt).astype('<f4').reshape(-1)
    assert vals.size == rows * cols, (qt, vals.size, rows, cols)
    out.write(struct.pack('<4i', int(qt), rows, cols, len(raw))); out.write(raw); out.write(vals.tobytes())

def synthetic(path):
    rng = np.random.default_rng(7)
    with open(path, 'wb') as out:
        for qt, (elems, bb) in SIZES.items():
            rows, cols = 3, 512
            blocks = rng.integers(0, 256, size=(rows * cols // elems, bb), dtype=np.uint8)
            def f16(n, lo, hi): return (rng.uniform(lo, hi, n) * rng.choice([-1, 1], n)).astype(np.float16).view(np.uint8).reshape(n, 2)
            n = blocks.shape[0]
            if qt in (T.Q4_0, T.Q8_0): blocks[:, 0:2] = f16(n, 1e-4, 0.1)
            elif qt == T.Q4_1: blocks[:, 0:2] = f16(n, 1e-4, 0.1); blocks[:, 2:4] = f16(n, 1e-3, 1)
            elif qt in (T.Q4_K, T.Q5_K): blocks[:, 0:2] = f16(n, 1e-5, 1e-2); blocks[:, 2:4] = f16(n, 1e-5, 1e-2)
            elif qt == T.Q6_K: blocks[:, 208:210] = f16(n, 1e-5, 1e-2)
            record(out, qt, blocks.tobytes(), rows, cols)

def real(path, models):
    seen = set()
    with open(path, 'wb') as out:
        for m in models:
            r = gguf.GGUFReader(m)
            for t in r.tensors:
                qt = t.tensor_type
                if qt not in SIZES or len(t.shape) != 2: continue
                key = (qt, m)
                if key in seen: continue
                seen.add(key)
                cols, rows_all = int(t.shape[0]), int(t.shape[1])
                rows = min(4, rows_all)
                elems, bb = SIZES[qt]
                raw = bytes(np.asarray(t.data).reshape(-1).view(np.uint8)[: rows * cols // elems * bb])
                record(out, qt, raw, rows, cols)
                print(m[-40:], t.name, qt.name, rows, 'x', cols)

if __name__ == '__main__':
    synthetic(sys.argv[1])
    if len(sys.argv) > 3: real(sys.argv[2], sys.argv[3:])
