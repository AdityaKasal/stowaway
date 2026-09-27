#!/usr/bin/env python3
"""Rewrite one GGUF file with its 8-bit always-needed weights at ~6 bits (Q8_0 -> Q5_1), freeing RAM for the expert
cache on small machines.

Qwen3.8-Flash-Next's 4-bit files keep attention and the other always-needed weights at 8 bits (3.9 GB). On an 8 GB
laptop those sit in RAM next to the expert cache, so every GB they take is a GB less of cached experts. This writes a
version of the file where those tensors are Q5_1 (6 bits per weight, ~30% smaller). Experts are left alone (they live
in the packed .bin), everything else keeps its bytes. GGUF needs tensors back to back in header order, so the offsets
after each shrunk tensor move; the header keeps its length because only fixed-width fields change.

Only tensors after the file's last big table are shrunk, so a 29 GB table never has to move. Because sizes only
shrink, every tensor moves down, and the file can be rewritten in place in one forward pass (a few GB of writes, no
second copy); a small journal lets an interrupted run pick up where it stopped.

usage: slim_dense.py <file.gguf> [--type q5_1|q5_0|q4_1]            (in place)
       slim_dense.py <in.gguf> --out <out.gguf> [...]                (a new file, for comparisons)
"""
import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "llama.cpp" / "gguf-py"))
from gguf import quants  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as T  # noqa: E402

import sparse  # noqa: E402

LEAVE = ("token_embd", "per_layer_token_embd", "output")  # lookup tables and the output layer (by name prefix)
MIN_BYTES = 256 << 10  # small tensors (norms, biases) are not worth it
BIG = 256 << 20  # nothing before the last tensor this big (that isn't an expert) is shrunk, so it never moves
TARGETS = {"q5_1": T.Q5_1, "q5_0": T.Q5_0, "q4_1": T.Q4_1}


class SimulatedCrash(Exception):
    """Raised by crash_after, to test that an interrupted run resumes correctly."""


def nbytes(typ, dims):
    bs, ts = GGML_QUANT_SIZES[T(typ)]
    ne = 1
    for d in dims:
        ne *= d
    return ne // bs * ts


def read_header(path):
    """(header bytes, [dict(name, dims, type, rel, type_pos, off_pos)], data_start, alignment)."""
    with open(path, "rb") as f:
        return parse_header(f.read(64 << 20))


def parse_header(buf):
    o = 0

    def rd(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, buf, o)[0]
        o += struct.calcsize("<" + fmt)
        return v

    def rstr():
        nonlocal o
        n = rd("Q")
        s = buf[o:o + n].decode("utf-8", errors="replace")
        o += n
        return s

    width = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    o = 4
    rd("I")
    n_tensors, n_kv = rd("Q"), rd("Q")
    align = 32
    for _ in range(n_kv):
        key, t = rstr(), rd("I")
        if t == 8:
            rstr()
        elif t == 9:
            at, n = rd("I"), rd("Q")
            if at == 8:
                for _ in range(n):
                    rstr()
            else:
                o += width[at] * n
        else:
            v = struct.unpack_from("<" + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q",
                                          11: "q", 12: "d"}[t], buf, o)[0]
            o += width[t]
            if key == "general.alignment":
                align = int(v)
    tensors = []
    for _ in range(n_tensors):
        name = rstr()
        dims = [rd("Q") for _ in range(rd("I"))]
        type_pos = o
        typ = rd("I")
        off_pos = o
        rel = rd("Q")
        tensors.append(dict(name=name, dims=dims, type=typ, rel=rel, type_pos=type_pos, off_pos=off_pos))
    data_start = (o + align - 1) // align * align
    return bytearray(buf[:data_start]), tensors, data_start, align


KEEP = ()  # name fragments never converted (for experiments: --keep hc_,shexp)


def convertible(t, min_bytes=MIN_BYTES):
    return (t["type"] == T.Q8_0 and "_exps." not in t["name"] and not t["name"].startswith(LEAVE)
            and not any(k in t["name"] for k in KEEP)
            and t["dims"][0] % 32 == 0 and nbytes(t["type"], t["dims"]) >= min_bytes)


def plan(header, tensors, align, target, min_bytes=MIN_BYTES):
    """Choose new types and offsets, and patch them into header. Returns (data size, bytes saved, tensors shrunk)."""
    last_big = max((i for i, t in enumerate(tensors)
                    if "_exps." not in t["name"] and nbytes(t["type"], t["dims"]) >= BIG), default=-1)
    pos = saved = n = 0
    for i, t in enumerate(tensors):
        t["new_type"] = target if i > last_big and convertible(t, min_bytes) else t["type"]
        t["new_rel"] = pos
        size = nbytes(t["new_type"], t["dims"])
        pos += (size + align - 1) // align * align
        if t["new_type"] != t["type"]:
            saved += nbytes(t["type"], t["dims"]) - size
            n += 1
        struct.pack_into("<I", header, t["type_pos"], int(t["new_type"]))
        struct.pack_into("<Q", header, t["off_pos"], t["new_rel"])
    return pos, saved, n


def requant(raw, t):
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(-1, nbytes(t["type"], t["dims"][:1]))
    return quants.quantize(quants.dequantize(rows, T(t["type"])), T(t["new_type"])).tobytes()


def padded(data, align):
    return data + b"\0" * (-len(data) % align)


def slim_copy(src, dst, target, min_bytes=MIN_BYTES, keep_experts=False):
    header, tensors, data_start, align = read_header(src)
    total, saved, n = plan(header, tensors, align, target, min_bytes)
    print(f"{Path(src).name}: {n} tensors to {T(target).name}, {saved / 1e9:.2f} GB smaller", flush=True)
    tmp = Path(str(dst) + ".part")
    sparse.make_sparse_file(tmp, data_start + total)
    t0 = time.time()
    with open(src, "rb") as fi, open(tmp, "r+b") as fo:
        fo.write(header)
        for t in tensors:
            if "_exps." in t["name"] and not keep_experts:
                continue  # a hole: the experts are read from the packed .bin
            size = nbytes(t["type"], t["dims"])
            fi.seek(data_start + t["rel"])
            fo.seek(data_start + t["new_rel"])
            if t["new_type"] != t["type"]:
                fo.write(padded(requant(fi.read(size), t), align))
            else:
                left = size
                while left:
                    chunk = fi.read(min(left, 64 << 20))
                    fo.write(chunk)
                    left -= len(chunk)
                fo.write(b"\0" * (-size % align))
    tmp.replace(dst)
    print(f"  wrote {dst} in {time.time() - t0:.0f} s", flush=True)


def _sync_write(path, data):
    tmp = Path(f"{path}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def pending(path, target=T.Q5_1, min_bytes=MIN_BYTES):
    """True if slim_in_place has work to do on this file (or an interrupted run to finish)."""
    header, tensors, data_start, align = read_header(path)
    return plan(header, tensors, align, target, min_bytes)[2] > 0 or Path(f"{path}.slim.json").exists()


def slim_in_place(path, target, min_bytes=MIN_BYTES, keep_experts=False, crash_after=None, quiet=False):
    """Shrink one file in place. Returns the bytes saved (0 if there was nothing to do)."""
    path = Path(path)
    jpath, spath = Path(f"{path}.slim.json"), Path(f"{path}.slim.src")
    header, tensors, data_start, align = read_header(path)
    total, saved, n = plan(header, tensors, align, target, min_bytes)
    state = json.loads(jpath.read_text()) if jpath.exists() else {"i": 0, "saved": -1}
    holes = [(data_start + t["new_rel"], nbytes(t["new_type"], t["dims"])) for t in tensors
             if "_exps." in t["name"] and not keep_experts]
    if not n:  # nothing to do, or only the clean-up of a run that was stopped after it rewrote the header
        if jpath.exists():
            os.truncate(path, data_start + total)
            if holes:
                sparse.punch(path, holes)
            spath.unlink(missing_ok=True)
            jpath.unlink()
        return 0
    if not quiet:
        print(f"{path.name}: {n} tensors to {T(target).name}, {saved / 1e9:.2f} GB smaller"
              + (f" (resuming at tensor {state['i']})" if state["i"] else ""), flush=True)
    moved, t0 = 0, time.time()
    with open(path, "r+b") as f:
        for i in range(state["i"], len(tensors)):
            t = tensors[i]
            size = nbytes(t["type"], t["dims"])
            src, dst = data_start + t["rel"], data_start + t["new_rel"]
            if ("_exps." in t["name"] and not keep_experts) or (src == dst and t["new_type"] == t["type"]):
                continue
            if state["saved"] == i and spath.exists():
                raw = spath.read_bytes()  # resumed: this tensor's own write may have overwritten part of it
            else:
                f.seek(src)
                raw = f.read(size)
            out = padded(requant(raw, t) if t["new_type"] != t["type"] else raw, align)
            if dst + len(out) > src and state["saved"] != i:  # the write overlaps its own source: keep a copy first
                _sync_write(spath, raw)
                state = {"i": i, "saved": i}
                _sync_write(jpath, json.dumps(state).encode())
            f.seek(dst)
            f.write(out)
            f.flush()
            os.fsync(f.fileno())
            state = {"i": i + 1, "saved": -1}
            _sync_write(jpath, json.dumps(state).encode())
            moved += 1
            if crash_after is not None and moved >= crash_after:
                raise SimulatedCrash(f"stopped after {moved} tensors")
        f.seek(0)
        f.write(header)
        f.flush()
        os.fsync(f.fileno())
    os.truncate(path, data_start + total)
    if holes:
        sparse.punch(path, holes)  # the expert ranges now hold leftovers of the moved tensors
    spath.unlink(missing_ok=True)
    jpath.unlink()
    if not quiet:
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", help="write a new file instead of changing src")
    ap.add_argument("--type", default="q5_1", choices=sorted(TARGETS))
    ap.add_argument("--min-kb", type=int, default=MIN_BYTES >> 10, help="skip tensors smaller than this")
    ap.add_argument("--keep-experts", action="store_true", help="copy the experts too (for a model that isn't packed)")
    ap.add_argument("--keep", default="", help="comma-separated name fragments to leave at 8 bits (e.g. hc_,shexp)")
    ap.add_argument("--crash-after", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args()
    KEEP = tuple(k for k in a.keep.split(",") if k)
    if a.out:
        slim_copy(a.src, a.out, TARGETS[a.type], a.min_kb << 10, a.keep_experts)
    else:
        slim_in_place(a.src, TARGETS[a.type], a.min_kb << 10, a.keep_experts, a.crash_after)
