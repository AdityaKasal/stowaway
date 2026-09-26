"""Write a copy of a model's routed experts with each expert's gate/up/down stored next to each other.

The GGUF stores each expert tensor (gate, up, down) as its own block of all 256 experts, so one expert is three
~2.3 MB pieces far apart. Reading them as one ~6.9 MB block is ~40% faster on the PC's SSD (read_latency.py).
The expert cache reads from this file when EXPERT_CACHE_PACKED points at it. With --slim the original model's copy of
the experts is freed layer by layer as it is packed (the file keeps its size and layout, the space is released).

Output: <out>.bin and <out>.idx, one line per expert tensor:
    <tensor name> <layer> <layer base offset> <block bytes> <offset of this tensor inside a block> <bytes>
Expert e of layer L starts at base + e * block. Everything is 4 KB aligned for unbuffered reads.

usage: python repack_experts.py <model-00001-of-0000N.gguf> <out> [--slim]
       python repack_experts.py <model.gguf> <out> --slim-after      (free the originals of an already packed model)
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


def _write_all(f, mv):
    """Unbuffered writes are capped (~2.1 GB per call on Linux and Windows) and may be partial: loop until done."""
    mv = memoryview(mv)
    while len(mv):
        n = f.write(mv[: 1 << 30])
        mv = mv[n:]


def _read_all(f, n):
    out = bytearray()
    while len(out) < n:
        chunk = f.read(min(n - len(out), 1 << 30))
        if not chunk:
            break
        out += chunk
    return bytes(out)


ORDER = {"ffn_gate_exps": 0, "ffn_up_exps": 1, "ffn_gate_up_exps": 1, "ffn_down_exps": 2}


def plan_layout(layers, n_expert):
    """The packed layout. layers: {layer: [(order, name, part index, file offset, bytes), ...]} for the expert weight
    tensors. Returns (plan, index lines, total bytes); plan = [(layer, base, block, [(name, part, file offset, bytes,
    offset in block, bytes per expert), ...]), ...]. Shared by repack() and the direct download (streampack.py), so
    both produce byte-identical files."""
    index_lines, base, plan = [], 0, []
    for L in sorted(layers):
        offs, block = [], 0
        for _, name, pi, foff, nbytes in sorted(layers[L]):
            per = nbytes // n_expert
            offs.append((name, pi, foff, nbytes, block, per))
            block += pad(per)
        plan.append((L, base, block, offs))
        for name, pi, foff, nbytes, off, per in offs:
            index_lines.append(f"{name} {L} {base} {block} {off} {per}")
        base += block * n_expert
    return plan, index_lines, base


def _readers(parts):
    return [gguf.GGUFReader(p) for p in parts]


def repack(first, out, slim=False):
    """Pack the experts into <out>.bin/.idx. slim=True also frees each layer's expert data in the original files once
    it is written and checked (sparse.punch), so the model needs ~1x its size on disk instead of 2x. Resumable: an
    interrupted run continues from <out>.progress (needed in slim mode, where finished layers are gone from the
    original)."""
    import gc
    import hashlib
    import os
    first, out = Path(first), Path(out)
    parts = sorted(first.parent.glob(first.name.replace("00001-of", "*-of"))) if "00001-of" in first.name else [first]
    meta = gguf.GGUFReader(parts[0])
    arch = bytes(meta.fields["general.architecture"].parts[-1]).decode()
    n_expert = int(meta.fields[f"{arch}.expert_count"].parts[-1][0])
    del meta

    # layer -> [(order, name, part index, file offset, bytes)] in a fixed order (gate, up, down)
    order = ORDER
    layers = defaultdict(list)
    readers = _readers(parts)
    for pi, r in enumerate(readers):
        for t in r.tensors:
            if "_exps.weight" in t.name:  # expert weights; per-expert biases (gpt-oss) stay in the model file
                kind = t.name.split(".")[2]
                layers[int(t.name.split(".")[1])].append((order[kind], t.name, pi, int(t.data_offset), int(t.n_bytes)))
    del readers, r, t
    gc.collect()

    plan, index_lines, total = plan_layout(layers, n_expert)

    progress = Path(f"{out}.progress")
    done_layers = set(int(x) for x in progress.read_text().split()) if progress.exists() else set()
    if done_layers:
        print(f"resuming: {len(done_layers)} of {len(plan)} layers already packed", flush=True)
    elif Path(f"{out}.bin").exists():
        Path(f"{out}.bin").unlink()
    print(f"{len(plan)} layers, {n_expert} experts, block {plan[0][2] / 1e6:.2f} MB, writing {total / 1e9:.1f} GB to "
          f"{out}.bin" + (" (freeing the originals' copy as it goes)" if slim else ""), flush=True)
    t0, done, freed = time.time(), 0, 0
    if not Path(f"{out}.bin").exists():
        import sparse  # sized without writing: on Windows, truncate() would zero-fill the whole file first
        sparse.make_sparse_file(f"{out}.bin", total)
    with open(f"{out}.bin", "r+b", buffering=0) as f:
        for L, lbase, block, offs in plan:
            if L in done_layers:
                continue
            # build a whole layer in memory then write it (1.8 GB for the 122B Q5); keeps writes large and sequential
            buf = bytearray(block * n_expert)
            mv = memoryview(buf)
            readers = _readers(parts)
            for name, pi, foff, nbytes, off, per in offs:
                t = next(x for x in readers[pi].tensors if x.name == name)
                raw = t.data.reshape(n_expert, -1).view("uint8").reshape(n_expert, -1)
                assert raw.shape[1] == per, (name, raw.shape, per)
                if slim:  # a slimmed layer from an earlier run reads back as zeros: refuse rather than pack zeros
                    assert raw[0].any() or raw[n_expert // 2].any(), f"{name} in the original is empty; can't pack it"
                for e in range(n_expert):
                    s = e * block + off
                    mv[s : s + per] = raw[e].tobytes()
            del readers, t, raw
            gc.collect()  # drop the original files' mappings (Windows can't free space in a mapped file)
            f.seek(lbase)
            _write_all(f, mv)
            os.fsync(f.fileno())
            # check what landed on disk against what we built, for a few experts spread over the layer
            for e in sorted({0, n_expert - 1, *range(1, n_expert, max(1, n_expert // 8))}):
                f.seek(lbase + e * block)
                got = _read_all(f, block)
                assert hashlib.sha1(got).digest() == hashlib.sha1(mv[e * block : (e + 1) * block]).digest(), \
                    f"layer {L} expert {e} didn't read back correctly"
            if slim:
                import sparse
                by_part = defaultdict(list)
                for name, pi, foff, nbytes, off, per in offs:
                    by_part[pi].append((foff, nbytes))
                for pi, ranges in by_part.items():
                    freed += sparse.punch(parts[pi], ranges)
            with open(progress, "a") as pf:
                pf.write(f"{L}\n")
            done += len(buf)
            el = time.time() - t0
            print(f"  layer {L:2d} done, {done / 1e9:6.1f} GB, {done / 1e9 / el:.2f} GB/s"
                  + (f", {freed / 1e9:.1f} GB freed in the original" if slim else ""), flush=True)
            del buf, mv
    Path(f"{out}.idx").write_text("\n".join(index_lines) + "\n")
    progress.unlink(missing_ok=True)
    print(f"wrote {out}.bin ({total / 1e9:.1f} GB) and {out}.idx in {time.time() - t0:.0f} s"
          + (f"; freed {freed / 1e9:.1f} GB in the original" if slim else ""))
    if not slim:
        # spot-check: random experts must match the original byte for byte
        import random
        rng = random.Random(0)
        readers = _readers(parts)
        with open(f"{out}.bin", "rb") as f:
            for _ in range(64):
                L, lbase, block, offs = rng.choice(plan)
                name, pi, foff, nbytes, off, per = rng.choice(offs)
                e = rng.randrange(n_expert)
                f.seek(lbase + e * block + off)
                t = next(x for x in readers[pi].tensors if x.name == name)
                assert _read_all(f, per) == t.data.reshape(n_expert, -1)[e].tobytes(), f"mismatch in {name} expert {e}"
        print("spot-check: 64 random experts match the original")


def slim_after(first, out):
    """Free the expert data in the original files of a model packed earlier (without slim)."""
    import sparse
    first = Path(first)
    parts = sorted(first.parent.glob(first.name.replace("00001-of", "*-of"))) if "00001-of" in first.name else [first]
    assert Path(f"{out}.idx").exists() and not Path(f"{out}.progress").exists(), "pack the experts first"
    freed = 0
    for p in parts:
        r = gguf.GGUFReader(p)
        ranges = [(int(t.data_offset), int(t.n_bytes)) for t in r.tensors if "_exps.weight" in t.name]
        del r
        import gc
        gc.collect()
        if ranges:
            freed += sparse.punch(p, ranges)
    print(f"freed {freed / 1e9:.1f} GB in the original model files")
    return freed


def main():
    if len(sys.argv) > 3 and sys.argv[3] == "--slim-after":
        slim_after(sys.argv[1], sys.argv[2])
    else:
        repack(sys.argv[1], sys.argv[2], slim="--slim" in sys.argv[3:])


if __name__ == "__main__":
    main()
