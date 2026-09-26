"""Download a Mixture-of-Experts GGUF straight into stowaway's layout: every byte is written once, to where it ends up.

The usual route downloads the whole model, then re-reads it and writes the experts a second time in the packed
layout (repack_experts.py), then frees the originals. Here the GGUF's header is read first (a few MB), which says where
every tensor lives; the download is then routed as it arrives: expert weights go straight into <packed>.bin in the
packed layout, and everything else goes into the .gguf at its own offset. The expert ranges of the .gguf are never
written, so they stay holes, exactly like a slimmed model. That halves the SSD writes and removes the setup step.

The result is byte-identical to downloading and then packing with --slim. The SHA-256 of the stream is checked
against Hugging Face's. Interrupted downloads resume from the last checkpoint (the hash is rebuilt by reading back what
was already written).

    fetch_packed(repo, files, into, packed, expected, user_agent) -> Path of the first .gguf, or None if the file isn't
    a MoE GGUF this can handle (the caller then downloads normally).
"""

import hashlib
import json
import struct
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import repack_experts
import sparse
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

HF = "https://huggingface.co"
CHUNK = 4 << 20
CHECKPOINT = 256 << 20


def _get(url, ua, start=0, end=None):
    rng = f"bytes={start}-" + ("" if end is None else str(end))
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": rng, "User-Agent": ua}), timeout=60)


def parse_header(buf):
    """(kv, [(name, n_bytes, relative offset)], data_start) from the first bytes of a GGUF file."""
    o = [0]

    def rd(fmt):
        v = struct.unpack_from("<" + fmt, buf, o[0])
        o[0] += struct.calcsize("<" + fmt)
        return v[0]

    def rstr():
        n = rd("Q")
        if o[0] + n > len(buf):
            raise EOFError
        v = buf[o[0]:o[0] + n].decode("utf-8", errors="replace")
        o[0] += n
        return v

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmts = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def val(t):
        if t == 8:
            return rstr()
        if t == 9:
            at, n = rd("I"), rd("Q")
            if at == 8:
                for _ in range(n):
                    rstr()
            else:
                o[0] += sizes[at] * n
            return None
        return rd(fmts[t])

    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    o[0] = 4
    rd("I")
    n_tensors, n_kv = rd("Q"), rd("Q")
    kv = {}
    for _ in range(n_kv):
        k = rstr()
        kv[k] = val(rd("I"))
    tensors = []
    for _ in range(n_tensors):
        name = rstr()
        dims = [rd("Q") for _ in range(rd("I"))]
        typ, rel = rd("I"), rd("Q")
        ne = 1
        for d in dims:
            ne *= d
        bs, ts = GGML_QUANT_SIZES[GGMLQuantizationType(typ)]
        tensors.append((name, ne // bs * ts, rel))
    align = int(kv.get("general.alignment") or 32)
    data_start = (o[0] + align - 1) // align * align
    return kv, tensors, data_start


def read_header(url, ua):
    n = 4 << 20
    while True:
        with _get(url, ua, 0, n - 1) as r:
            buf = r.read()
        try:
            return parse_header(buf)
        except (struct.error, EOFError, IndexError):
            if len(buf) < n or n >= 512 << 20:
                raise
            n *= 4


class Router:
    """Maps a byte range of one part's stream to (file, offset) pieces: expert weights -> packed, the rest -> gguf."""

    def __init__(self, experts):
        # experts: sorted [(abs start, n_bytes, layer base, block, offset in block, bytes per expert)]
        self.experts = experts

    def pieces(self, a, b):
        """Yield (is_expert, dst offset, src start, src end) covering stream bytes [a, b), in order."""
        pos = a
        for (t0, n, lbase, block, off, per) in self.experts:
            t1 = t0 + n
            if t1 <= pos:
                continue
            if t0 >= b:
                break
            if pos < t0:
                yield (False, pos, pos, t0)
                pos = t0
            end = min(b, t1)
            while pos < end:
                e, within = divmod(pos - t0, per)
                step = min(end - pos, per - within)
                yield (True, lbase + e * block + off + within, pos, pos + step)
                pos += step
        if pos < b:
            yield (False, pos, pos, b)


def fetch_packed(repo, files, into, packed, expected, ua):
    into, packed = Path(into), Path(packed)
    urls = [f"{HF}/{repo}/resolve/main/{f}" for f in files]
    headers = [read_header(u, ua) for u in urls]
    kv0 = headers[0][0]
    arch = kv0.get("general.architecture")
    n_expert = kv0.get(f"{arch}.expert_count")
    if not n_expert:
        return None
    layers = defaultdict(list)
    for pi, (kv, tensors, data_start) in enumerate(headers):
        for name, nb, rel in tensors:
            if "_exps.weight" in name:
                layers[int(name.split(".")[1])].append((repack_experts.ORDER[name.split(".")[2]], name, pi,
                                                        data_start + rel, nb))
    if not layers:
        return None
    plan, index_lines, total = repack_experts.plan_layout(layers, n_expert)
    routes = defaultdict(list)
    for L, lbase, block, offs in plan:
        for name, pi, foff, nb, off, per in offs:
            routes[pi].append((foff, nb, lbase, block, off, per))
    routers = {pi: Router(sorted(v)) for pi, v in routes.items()}

    state_path = Path(f"{packed}.stream.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"part": 0, "done": 0}
    into.mkdir(parents=True, exist_ok=True)
    bin_path = Path(f"{packed}.bin")
    if not state_path.exists() or not bin_path.exists():
        sparse.make_sparse_file(bin_path, total)  # written in scattered order: must not be zero-filled first
        state = {"part": 0, "done": 0}
        state_path.write_text(json.dumps(state))
    print(f"downloading straight into the fast layout: {total / 1e9:.1f} GB of experts + the rest of the model "
          f"(no separate setup step afterwards)", flush=True)

    with open(bin_path, "r+b") as fb:
        for pi, (f, url) in enumerate(zip(files, urls)):
            final = into / Path(f).name
            if pi < state["part"] or final.exists():
                continue
            part = Path(str(final) + ".part")
            size, sha = expected[pi] if expected and expected[pi] else (None, None)
            if size is None:
                with _get(url, ua, 0, 0) as r:
                    size = int(r.headers["Content-Range"].split("/")[-1])
            if state.get("part") != pi or state["done"] == 0 or not part.exists():
                sparse.make_sparse_file(part, size)  # the expert ranges are never written: they stay holes
                state = {"part": pi, "done": 0}
                state_path.write_text(json.dumps(state))
            router = routers.get(pi, Router([]))
            h = hashlib.sha256()
            with open(part, "r+b") as fg:
                if state["done"]:  # resuming: rebuild the hash from what's already on disk
                    print(f"  resuming {final.name} at {state['done'] / 1e9:.1f} GB (re-checking what's there)", flush=True)
                    for is_exp, dst, a, b in router.pieces(0, state["done"]):
                        src = fb if is_exp else fg
                        src.seek(dst)
                        left = b - a
                        while left:
                            chunk = src.read(min(left, 16 << 20))
                            h.update(chunk)
                            left -= len(chunk)
                done, t0, last, since_ck = state["done"], time.time(), 0.0, 0
                attempt = 0
                while done < size:
                    try:
                        with _get(url, ua, done) as r:
                            while done < size:
                                data = r.read(CHUNK)
                                if not data:
                                    break
                                h.update(data)
                                mv = memoryview(data)
                                for is_exp, dst, a, b in router.pieces(done, done + len(data)):
                                    dstf = fb if is_exp else fg
                                    dstf.seek(dst)
                                    dstf.write(mv[a - done:b - done])
                                done += len(data)
                                since_ck += len(data)
                                if since_ck >= CHECKPOINT:
                                    fb.flush()
                                    fg.flush()
                                    state.update(part=pi, done=done)
                                    state_path.write_text(json.dumps(state))
                                    since_ck = 0
                                if time.time() - last > 2:
                                    last = time.time()
                                    rate = (done - state["done"]) / max(time.time() - t0, 1e-3) / 1e6
                                    print(f"\r  {final.name}: {done / 1e9:.1f} / {size / 1e9:.1f} GB  ({rate:.0f} MB/s)   ",
                                          end="", flush=True)
                    except Exception as e:  # network hiccup: continue from the last checkpoint
                        attempt += 1
                        if attempt > 20:
                            raise
                        print(f"\n  download interrupted ({e}); resuming...", flush=True)
                        fb.flush()
                        fg.flush()
                        done = state["done"]
                        h = hashlib.sha256()
                        for is_exp, dst, a, b in router.pieces(0, done):
                            src = fb if is_exp else fg
                            src.seek(dst)
                            left = b - a
                            while left:
                                chunk = src.read(min(left, 16 << 20))
                                h.update(chunk)
                                left -= len(chunk)
                        time.sleep(min(30, 2 + attempt * 3))
                fb.flush()
                fg.flush()
            print(f"\r  {final.name}: done ({size / 1e9:.1f} GB)" + " " * 20, flush=True)
            if sha and h.hexdigest() != sha:
                part.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
                bin_path.unlink(missing_ok=True)
                sys.exit(f"{final.name} arrived damaged (its checksum doesn't match Hugging Face's). "
                         "Check your connection and disk, then run stowaway again to re-download.")
            if not sha:
                print("  (couldn't get the file's checksum from Hugging Face, so skipping the integrity check)")
            part.rename(final)
            state = {"part": pi + 1, "done": 0}
            state_path.write_text(json.dumps(state))
    Path(f"{packed}.idx").write_text("\n".join(index_lines) + "\n")
    state_path.unlink(missing_ok=True)
    return into / Path(files[0]).name
