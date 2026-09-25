"""Write the always-needed ("dense") matmul weights of a model layer by layer into one file, so moe-stream can read a
whole streamed layer with one big read instead of a dozen scattered ones (EXPERT_CACHE_DENSE_PACKED).

Same selection as moe-stream.h: every non-expert tensor of at least 256 KB except the token embedding table.
Each tensor starts on a 4 KB boundary (unbuffered reads). Output: <out>.bin and <out>.idx ("<name> <offset> <bytes>").

usage: python pack_dense.py models/122b/Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf models/122b/dense-packed
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).parent / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

ALIGN = 4096


def pack(first, out):
    first, out = Path(first), str(out)
    parts = sorted(first.parent.glob(first.name.replace("00001-of", "*-of"))) if "00001-of" in first.name else [first]
    by_layer = defaultdict(list)
    for p in parts:
        for t in gguf.GGUFReader(p).tensors:
            if "_exps." in t.name or t.name.startswith("token_embd") or int(t.n_bytes) < (256 << 10):
                continue
            layer = int(t.name.split(".")[1]) if t.name.startswith("blk.") else 1 << 20  # output layer etc. last
            by_layer[layer].append(t)
    lines, pos, t0 = [], 0, time.time()
    with open(f"{out}.bin", "wb") as f:
        for L in sorted(by_layer):
            for t in by_layer[L]:
                pad = (-pos) % ALIGN
                f.write(b"\0" * pad)
                pos += pad
                data = t.data.tobytes()
                assert len(data) == int(t.n_bytes)
                f.write(data)
                lines.append(f"{t.name} {pos} {len(data)}")
                pos += len(data)
        f.write(b"\0" * ((-pos) % ALIGN))  # so the last aligned read stays inside the file
    Path(f"{out}.idx").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}.bin ({pos / 1e9:.2f} GB, {len(lines)} tensors) in {time.time() - t0:.0f} s")


def main():
    pack(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
