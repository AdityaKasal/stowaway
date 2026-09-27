#!/usr/bin/env python3
"""Experiment: take a file's always-needed weights from another quant of the same model instead of rounding them
ourselves. Unsloth's UD-Q2_K_XL of Qwen3.8-Flash-Next stores attention and DeltaNet weights as Q5_K/Q6_K made with an
importance matrix (and keeps the sensitive hyper-connection weights at 8 bits); slim_dense.py's plain rounding to
Q5_1 cost too much (RESULTS.md 29). Only the chosen tensors are fetched (HTTP range requests), the rest is copied.

usage: splice_dense.py <our file.gguf> <out.gguf> <repo> <donor folder> [--kinds attn_qkv,attn_gate,...]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slim_dense as sd  # noqa: E402
import sparse  # noqa: E402
import streampack  # noqa: E402

UA = "stowaway-research"


def donor_tensors(repo, folder):
    """name -> (url, absolute offset, type, dims) for every tensor in the donor quant's files."""
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/tree/main/{folder}", headers={"User-Agent": UA})
    files = sorted(f["path"] for f in json.load(urllib.request.urlopen(req)) if f["path"].endswith(".gguf"))
    out = {}
    for f in files:
        url = f"https://huggingface.co/{repo}/resolve/main/{f}"
        n = 16 << 20
        while True:
            with streampack._get(url, UA, 0, n - 1) as r:
                buf = r.read()
            try:
                _, tensors, data_start, _ = sd.parse_header(buf)
                break
            except Exception:
                if len(buf) < n:
                    raise
                n *= 4
        for t in tensors:
            out[t["name"]] = (url, data_start + t["rel"], t["type"], t["dims"])
    return out


def splice(src, dst, repo, folder, kinds):
    header, tensors, data_start, align = sd.read_header(src)
    donor = donor_tensors(repo, folder)
    pos = saved = n = 0
    for t in tensors:
        d = donor.get(t["name"])
        use = (d is not None and "_exps." not in t["name"] and not t["name"].startswith(sd.LEAVE)
               and (not kinds or any(k in t["name"] for k in kinds)) and list(d[3]) == list(t["dims"])
               and sd.nbytes(d[2], t["dims"]) < sd.nbytes(t["type"], t["dims"]))
        t["donor"] = d if use else None
        t["new_type"] = d[2] if use else t["type"]
        t["new_rel"] = pos
        size = sd.nbytes(t["new_type"], t["dims"])
        pos += (size + align - 1) // align * align
        if use:
            saved += sd.nbytes(t["type"], t["dims"]) - size
            n += 1
        sd.struct.pack_into("<I", header, t["type_pos"], int(t["new_type"]))
        sd.struct.pack_into("<Q", header, t["off_pos"], t["new_rel"])
    kinds_used = sorted({sd.T(t["new_type"]).name for t in tensors if t["donor"]})
    print(f"{Path(src).name}: {n} tensors from {folder} ({', '.join(kinds_used)}), {saved / 1e9:.2f} GB smaller", flush=True)
    tmp = Path(f"{dst}.part")
    sparse.make_sparse_file(tmp, data_start + pos)
    t0 = time.time()
    with open(src, "rb") as fi, open(tmp, "r+b") as fo:
        fo.write(header)
        for t in tensors:
            if "_exps." in t["name"]:
                continue
            fo.seek(data_start + t["new_rel"])
            if t["donor"]:
                url, off, typ, dims = t["donor"]
                size = sd.nbytes(typ, dims)
                with streampack._get(url, UA, off, off + size - 1) as r:
                    data = r.read()
                assert len(data) == size, (t["name"], len(data), size)
            else:
                fi.seek(data_start + t["rel"])
                data = fi.read(sd.nbytes(t["type"], t["dims"]))
            fo.write(sd.padded(data, align))
    tmp.replace(dst)
    print(f"  wrote {dst} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("repo")
    ap.add_argument("folder")
    ap.add_argument("--kinds", default="", help="comma-separated name fragments to take from the donor (default: all)")
    a = ap.parse_args()
    splice(a.src, a.dst, a.repo, a.folder, [k for k in a.kinds.split(",") if k])
