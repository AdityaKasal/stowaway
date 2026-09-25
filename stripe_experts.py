"""Copy part of a packed expert file to a second drive, so the cache can read from both drives at once.

Two modes:
  <MOD> <REM>   copy whole experts e with e % MOD == REM (helps prompts: many experts load at once)
  tail <FRAC>   copy the last FRAC of every expert block; the cache reads the head from the main drive and the tail
                from this one in parallel, so even a single expert loads faster (helps generation too)

Reads <packed>.bin/.idx (from repack_experts.py). Writes <out>.bin with, per layer, the chosen experts back to back,
and <out>.stripe:
    first line:  <MOD> <REM>
    then:        <layer> <offset of that layer's first copied expert in out.bin> <block bytes>
Expert e (e % MOD == REM) of layer L is at offset_L + (e // MOD) * block.

usage: python stripe_experts.py models/122b/experts-packed E:/moe/experts-stripe 3 1
"""

import sys
import time
from pathlib import Path


def tail_mode(packed, out, frac):
    """<out>.stripe: first line "tail"; then <layer> <offset in out.bin> <block bytes> <split>.
    Expert e's bytes [split, block) are at offset + e * (block - split)."""
    layers = {}
    for line in Path(f"{packed}.idx").read_text().split("\n"):
        if line.strip():
            name, L, base, block, off, per = line.split()
            layers[int(L)] = (int(base), int(block))
    n_expert = 256
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    lines, pos, t0 = ["tail"], 0, time.time()
    with open(f"{packed}.bin", "rb", buffering=0) as src, open(f"{out}.bin", "wb", buffering=0) as dst:
        for L in sorted(layers):
            base, block = layers[L]
            split = int(block * (1 - frac)) // 4096 * 4096
            lines.append(f"{L} {pos} {block} {split}")
            chunk = bytearray()
            for e in range(n_expert):
                src.seek(base + e * block + split)
                chunk += src.read(block - split)
            dst.write(chunk)
            pos += len(chunk)
            print(f"  layer {L:2d}: {pos / 1e9:6.1f} GB, {pos / 1e9 / (time.time() - t0):.2f} GB/s", flush=True)
    Path(f"{out}.stripe").write_text("\n".join(lines) + "\n")
    import random
    rng = random.Random(1)
    info = {int(l.split()[0]): (int(l.split()[1]), int(l.split()[3])) for l in lines[1:]}
    with open(f"{packed}.bin", "rb") as src, open(f"{out}.bin", "rb") as dst:
        for _ in range(32):
            L = rng.choice(sorted(layers))
            base, block = layers[L]
            off2, split = info[L]
            e = rng.randrange(n_expert)
            src.seek(base + e * block + split)
            dst.seek(off2 + e * (block - split))
            assert src.read(block - split) == dst.read(block - split), f"mismatch layer {L} expert {e}"
    print(f"wrote {out}.bin ({pos / 1e9:.1f} GB) in {time.time() - t0:.0f} s; spot-check of 32 experts passed")


def main():
    packed, out = sys.argv[1], sys.argv[2]
    if sys.argv[3] == "tail":
        return tail_mode(packed, out, float(sys.argv[4]))
    mod, rem = int(sys.argv[3]), int(sys.argv[4])
    layers = {}
    for line in Path(f"{packed}.idx").read_text().split("\n"):
        if line.strip():
            name, L, base, block, off, per = line.split()
            layers[int(L)] = (int(base), int(block))
    n_expert = 256
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    lines, pos, t0 = [f"{mod} {rem}"], 0, time.time()
    with open(f"{packed}.bin", "rb", buffering=0) as src, open(f"{out}.bin", "wb", buffering=0) as dst:
        for L in sorted(layers):
            base, block = layers[L]
            lines.append(f"{L} {pos} {block}")
            chunk = bytearray()
            for e in range(rem, n_expert, mod):
                src.seek(base + e * block)
                chunk += src.read(block)
            dst.write(chunk)
            pos += len(chunk)
            print(f"  layer {L:2d}: {pos / 1e9:6.1f} GB, {pos / 1e9 / (time.time() - t0):.2f} GB/s", flush=True)
    Path(f"{out}.stripe").write_text("\n".join(lines) + "\n")

    # spot-check a few experts against the source
    import random
    rng = random.Random(1)
    with open(f"{packed}.bin", "rb") as src, open(f"{out}.bin", "rb") as dst:
        offs = {int(l.split()[0]): int(l.split()[1]) for l in lines[1:]}
        for _ in range(32):
            L = rng.choice(sorted(layers))
            base, block = layers[L]
            e = rng.randrange(rem, n_expert, mod)
            src.seek(base + e * block)
            dst.seek(offs[L] + (e // mod) * block)
            assert src.read(block) == dst.read(block), f"mismatch layer {L} expert {e}"
    print(f"wrote {out}.bin ({pos / 1e9:.1f} GB) in {time.time() - t0:.0f} s; spot-check of 32 experts passed")


if __name__ == "__main__":
    main()
