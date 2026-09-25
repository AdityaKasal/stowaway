"""Write a copy of a model's routed experts with each expert's gate/up/down stored next to each other.

The GGUF stores each expert tensor (gate, up, down) as its own block of all 256 experts, so one expert is three
~2.3 MB pieces far apart. Reading them as one ~6.9 MB block is ~40% faster on the PC's SSD (read_latency.py).
The original model is untouched; the expert cache reads from this file when EXPERT_CACHE_PACKED points at it.

Output: <out>.bin and <out>.idx, one line per expert tensor:
    <tensor name> <layer> <layer base offset> <block bytes> <offset of this tensor inside a block> <bytes>
Expert e of layer L starts at base + e * block. Everything is 4 KB aligned for unbuffered reads.

usage: python repack_experts.py models/122b/Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf models/122b/experts-packed
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).parent / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

ALIGN = 4096


def pad(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def repack(first, out):
    first, out = Path(first), Path(out)
    parts = sorted(first.parent.glob(first.name.replace("00001-of", "*-of"))) if "00001-of" in first.name else [first]
    meta = gguf.GGUFReader(parts[0])
    arch = bytes(meta.fields["general.architecture"].parts[-1]).decode()
    n_expert = int(meta.fields[f"{arch}.expert_count"].parts[-1][0])

    # layer -> [(name, tensor)] in a fixed order (gate, up, down)
    order = {"ffn_gate_exps": 0, "ffn_up_exps": 1, "ffn_gate_up_exps": 1, "ffn_down_exps": 2}
    layers = defaultdict(list)
    readers = [gguf.GGUFReader(p) for p in parts]
    for r in readers:
        for t in r.tensors:
            if "_exps." in t.name:
                kind = t.name.split(".")[2]
                layers[int(t.name.split(".")[1])].append((order[kind], t.name, t))

    index_lines, base = [], 0
    plan = []
    for L in sorted(layers):
        items = sorted(layers[L])
        offs, block = [], 0
        for _, name, t in items:
            per = int(t.n_bytes) // n_expert
            offs.append((name, t, block, per))
            block += pad(per)
        plan.append((L, base, block, offs))
        for name, t, off, per in offs:
            index_lines.append(f"{name} {L} {base} {block} {off} {per}")
        base += block * n_expert

    total = base
    print(f"{len(plan)} layers, {n_expert} experts, block {plan[0][2] / 1e6:.2f} MB, writing {total / 1e9:.1f} GB to {out}.bin", flush=True)
    t0, done = time.time(), 0
    with open(f"{out}.bin", "wb", buffering=0) as f:
        for L, lbase, block, offs in plan:
            # build a whole layer in memory then write it (1.8 GB for the 122B); keeps writes large and sequential
            buf = bytearray(block * n_expert)
            mv = memoryview(buf)
            for name, t, off, per in offs:
                raw = t.data.reshape(n_expert, -1).view("uint8").reshape(n_expert, -1)
                assert raw.shape[1] == per, (name, raw.shape, per)
                for e in range(n_expert):
                    s = e * block + off
                    mv[s : s + per] = raw[e].tobytes()
            f.write(buf)
            done += len(buf)
            el = time.time() - t0
            print(f"  layer {L:2d} done, {done / 1e9:6.1f} GB, {done / 1e9 / el:.2f} GB/s", flush=True)
    Path(f"{out}.idx").write_text("\n".join(index_lines) + "\n")
    print(f"wrote {out}.bin ({total / 1e9:.1f} GB) and {out}.idx in {time.time() - t0:.0f} s")

    # spot-check: random experts must match the original byte for byte
    import random
    rng = random.Random(0)
    with open(f"{out}.bin", "rb") as f:
        for _ in range(64):
            L, lbase, block, offs = rng.choice(plan)
            name, t, off, per = rng.choice(offs)
            e = rng.randrange(n_expert)
            f.seek(lbase + e * block + off)
            got = f.read(per)
            want = t.data.reshape(n_expert, -1)[e].tobytes()
            assert got == want, f"mismatch in {name} expert {e}"
    print("spot-check: 64 random experts match the original")


def main():
    repack(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
