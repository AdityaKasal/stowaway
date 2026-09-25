"""stowaway: run big Mixture-of-Experts language models on ordinary computers - no GPU, little RAM.

    stowaway list                     models it can download for you
    stowaway run qwen3.5-35b          download (asks first), set up once, chat in your browser
    stowaway run path/to/model.gguf   or run a MoE model you already have
    stowaway run qwen3.5-122b --cli   chat in the terminal instead
    stowaway run qwen3.5-35b --fast   ~1.5x less reading from disk; answers differ slightly from the full model
    stowaway run qwen3.5-35b -p "Hi"  answer one prompt and exit
    stowaway plan qwen3.5-122b        show the memory plan and expected speed, then stop

It checks free RAM and drive speed, packs the model's experts once (a copy laid out for fast reads; for models it
downloaded, the model file's own copy of the experts is freed as it goes, so a model needs about its own size on disk),
sizes the caches to fit, and starts llama.cpp with moe-stream turned on.
"""

import argparse
import math
import ctypes
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)  # running as the bundled program (PyInstaller)
HERE = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
if not FROZEN:
    sys.path.insert(0, str(HERE / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

VERSION = "0.2.6"
REPO = "AdityaKasal/stowaway"

import pack_dense  # noqa: E402
import repack_experts  # noqa: E402

GB = 1e9
MARGIN_GB = 1.0          # left free so the machine stays usable
BASE_GB = 1.2            # llama.cpp's own buffers (-b 128, 4k context), measured
MIN_EXPERT_CACHE_GB = 0.5
# Small machines (under 5 GB free, e.g. 4 GB laptops): 2k context and batch 64 need ~0.5 GB of buffers (measured peak
# 1.36 GB with a 0.3 GB cache and 0.8 GB of streamed weights, 35B, 4 GB VM), and every 100 MB matters. Leaving the
# always-needed weights to the OS there is fragile (3.1 tok/s in one test, 0.4 in the app with its own process added),
# so small machines stream them unless there is a clear surplus (slack 0.6). The 1 GB margin stays: with less than
# ~1.3 GB truly free, the OS evicts llama.cpp's small mapped tensors and refaults them (0.6 tok/s at 0.45 GB free vs
# 1.8 at 1.4 GB free, same model and VM).
SMALL_RAM_GB = 5.0
SMALL = {"margin": 1.0, "base": 0.5, "min_cache": 0.2, "slack": 0.6, "ctx": 2048, "batch": 64}
BIG = {"margin": MARGIN_GB, "base": BASE_GB, "min_cache": MIN_EXPERT_CACHE_GB, "slack": 0.3, "ctx": 4096, "batch": 128}
DRAFT_OVERHEAD_GB = 0.15 # helper model's context and buffers, on top of its file size
# Speculative decoding: something cheap guesses a few tokens, the big model checks them in one pass. It only pays off
# when the always-needed weights are streamed (one pass reads them once for several tokens); when they fit in RAM,
# the extra tokens only add expert reads (RESULTS.md sections 11 and 14). Same model and quality, but llama.cpp's
# batched check rounds differently, so an occasional word can differ from a run without it.
SPEC_FLAGS = ["--spec-type", "draft-simple", "--spec-draft-n-max", "12", "--spec-draft-p-min", "0.8", "-ngld", "0"]
MTP_FLAGS = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.5"]


# ---------------------------------------------------------------- machine

def available_ram_gb():
    if platform.system() == "Windows":
        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                        ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
        m = MS()
        m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail / GB
    if platform.system() == "Darwin":
        page = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.pagesize"]))
        vm = subprocess.check_output(["/usr/bin/vm_stat"]).decode()
        get = lambda k: int(next(l for l in vm.splitlines() if l.startswith(k)).split(":")[1].strip(" ."))  # noqa: E731
        return (get("Pages free") + get("Pages inactive") + get("Pages purgeable")) * page / GB
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / GB
    return 4.0


def drive_speed_gbps(path, seconds=2.0, threads=4):
    """Rough read speed at random spots of a big file, several reads at once like the cache does."""
    import threading
    size = os.path.getsize(path)
    block = 8 << 20
    if size < 16 * block:
        return None
    total = [0]
    lock = threading.Lock()
    stop = time.time() + seconds

    def reader():
        n = 0
        with open(path, "rb", buffering=0) as f:
            while time.time() < stop:
                f.seek(random.randrange(0, size - block) // 4096 * 4096)
                n += len(f.read(block))
        with lock:
            total[0] += n

    t0 = time.time()
    ts = [threading.Thread(target=reader) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return total[0] / (time.time() - t0) / GB


# ---------------------------------------------------------------- model

def split_parts(first):
    first = Path(first)
    if "-00001-of-" in first.name:
        return sorted(first.parent.glob(first.name.replace("00001-of", "*-of")))
    return [first]


def model_info(first):
    parts = split_parts(first)
    meta = gguf.GGUFReader(parts[0])
    arch = bytes(meta.fields["general.architecture"].parts[-1]).decode()
    field = lambda k: int(meta.fields[f"{arch}.{k}"].parts[-1][0])  # noqa: E731
    info = {"arch": arch, "layers": field("block_count"), "parts": parts,
            "experts": field("expert_count") if f"{arch}.expert_count" in meta.fields else 0}
    info["k"] = field("expert_used_count") if info["experts"] else 0
    info["mtp"] = f"{arch}.nextn_predict_layers" in meta.fields and field("nextn_predict_layers") > 0
    exp = dense = dense_managed = embd = 0
    for p in parts:
        for t in gguf.GGUFReader(p).tensors:
            n = int(t.n_bytes)
            if "_exps." in t.name:
                exp += n
            elif t.name.startswith("token_embd"):
                embd += n
            else:
                dense += n
                if n >= 256 << 10:
                    dense_managed += n
    info.update(expert_gb=exp / GB, dense_gb=dense / GB, dense_managed_gb=dense_managed / GB, embd_gb=embd / GB)
    info["active_expert_gb"] = exp / GB * info["k"] / max(info["experts"], 1)
    info["file_gb"] = sum(p.stat().st_size for p in parts) / GB
    return info


def cache_hit_estimate(fraction):
    """LRU hit rate vs share of experts cached, from the router study (122B/35B traces)."""
    pts = [(0.0, 0.0), (0.02, 0.2), (0.08, 0.45), (0.15, 0.6), (0.2, 0.66), (0.3, 0.76), (0.5, 0.87), (0.6, 0.9),
           (0.8, 0.96), (1.0, 1.0)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if fraction <= x1:
            return y0 + (y1 - y0) * (fraction - x0) / (x1 - x0)
    return 1.0


# ---------------------------------------------------------------- plan

def make_plan(info, ram_gb, drive, draft_gb=0.0):
    k = SMALL if ram_gb < SMALL_RAM_GB else BIG
    budget = ram_gb - k["margin"] - k["base"] - draft_gb
    plan = {"budget_gb": budget, "ctx": k["ctx"], "batch": k["batch"], "small": k is SMALL}
    if budget <= 0:
        return None, f"only {ram_gb:.1f} GB of RAM is free; close some programs (need at least ~{k['margin'] + k['base'] + 1.2:.1f} GB)"
    dense_all = info["dense_gb"]
    if budget >= dense_all + k["min_cache"] + k["slack"]:
        # the always-needed weights fit: leave them to the OS, the rest goes to the expert cache
        plan["dense_stream_gb"] = 0
        plan["cache_gb"] = min(budget - dense_all - k["slack"], info["expert_gb"])
        plan["pregate"] = 6
        streamed_dense = 0
    else:
        # they don't fit: stream them too, with a small expert cache (what worked for the 122B on 8 GB)
        plan["cache_gb"] = k["min_cache"]
        plan["dense_stream_gb"] = budget - k["min_cache"]
        per_layer = info["dense_managed_gb"] / info["layers"]
        minimum = 4 * per_layer * 1.3 + (info["dense_managed_gb"] - per_layer * info["layers"]) + per_layer
        if plan["dense_stream_gb"] < minimum:
            need = math.ceil((minimum + k["min_cache"] + k["base"] + k["margin"] + draft_gb) * 10) / 10
            return None, f"not enough free RAM: this model needs at least ~{need:.1f} GB free, you have {ram_gb:.1f} GB"
        plan["pregate"] = 0
        streamed_dense = max(0.0, info["dense_managed_gb"] - (plan["dense_stream_gb"] - 4 * per_layer * 1.3))
    hit = cache_hit_estimate(plan["cache_gb"] / max(info["expert_gb"], 1e-9))
    per_token = info["active_expert_gb"] * (1 - hit) + streamed_dense
    plan["read_per_token_gb"] = per_token
    plan["est_tok_s"] = None if not drive else 1 / (per_token / drive + 0.1)
    if plan["est_tok_s"] and draft_gb:
        plan["est_tok_s"] *= 2.0 if plan["dense_stream_gb"] else 1.3  # measured gain from the helper model
    return plan, None


# ---------------------------------------------------------------- run

def free_port(port):
    """The requested port, or the next free one (e.g. when a chat is already running)."""
    import socket
    for p in range(port, port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            try:
                sk.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return port


def cpu_has_fast_path():
    """The normal engine needs AVX2 + FMA + F16C + BMI2 (Intel 2013+, AMD 2015+). Older or budget CPUs get the
    compatible (SSE4.2) engine from the compat/ folder. STOWAWAY_COMPAT=1 forces it."""
    if os.environ.get("STOWAWAY_COMPAT"):
        return False
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "x64"):
        return True  # Apple Silicon / ARM builds have no such split
    try:
        if platform.system() == "Windows":
            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))  # PF_AVX2_INSTRUCTIONS_AVAILABLE
        if platform.system() == "Linux":
            flags = next(l for l in open("/proc/cpuinfo") if l.startswith("flags")).split()
            return all(f in flags for f in ("avx2", "fma", "f16c", "bmi2"))
        if platform.system() == "Darwin":
            out = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.leaf7_features", "machdep.cpu.features"]).decode().upper()
            return "AVX2" in out and "FMA" in out and "F16C" in out and "BMI2" in out
    except Exception:
        pass
    return True


def find_bin(name, bin_dir):
    exe = name + (".exe" if platform.system() == "Windows" else "")
    if not cpu_has_fast_path():
        for d in ([Path(bin_dir)] if bin_dir else []) + [HERE]:
            if (d / "compat" / exe).exists():
                if not getattr(find_bin, "_told", False):
                    print("cpu:     this processor lacks AVX2, so using the compatible engine (works everywhere, slower)")
                    find_bin._told = True
                return d / "compat" / exe
    for d in ([Path(bin_dir)] if bin_dir else []) + [HERE, HERE / "llama.cpp" / "build" / "bin", HERE / "build" / "bin"]:
        if (d / exe).exists():
            return d / exe
    sys.exit(f"can't find {exe}; build llama.cpp first (see README) or pass --bin")


def find_draft(first, arg):
    """A small model of the same family next to the big one (e.g. Qwen3.5-0.8B-*.gguf), or --draft."""
    if arg:
        if arg == "none":
            return None
        if not Path(arg).exists():
            sys.exit(f"helper model not found: {arg}")
        return Path(arg)
    family = first.name.split("-")[0]  # "Qwen3.5"
    for d in {first.parent, first.parent.parent}:
        for c in sorted(d.glob(f"{family}-0.8B*.gguf")) + sorted(d.glob(f"{family}-2B*.gguf")):
            return c
    return None


def ensure_dense_packed(info, dense_packed):
    if Path(f"{dense_packed}.idx").exists() and Path(f"{dense_packed}.bin").exists():
        return
    print(f"one-time setup: packing {info['dense_gb']:.1f} GB of always-needed weights layer by layer...", flush=True)
    pack_dense.pack(info["parts"][0], dense_packed)


def ensure_packed(info, packed, slim):
    """Pack the experts once. slim: free the original file's copy of them layer by layer (the model then needs ~1x
    its size on disk instead of 2x); only for models stowaway downloaded itself, or when asked with --slim."""
    if Path(f"{packed}.idx").exists() and Path(f"{packed}.bin").exists() and not Path(f"{packed}.progress").exists():
        return
    resuming = Path(f"{packed}.progress").exists()
    free = shutil.disk_usage(Path(packed).parent).free / GB
    need = (info["expert_gb"] / max(info["layers"], 1) * 1.2 + 2) if slim else info["expert_gb"] + 5
    if free < need and not resuming:
        sys.exit(f"packing needs {need:.0f} GB free next to the model; only {free:.0f} GB free")
    print(f"one-time setup: packing {info['expert_gb']:.1f} GB of experts for fast reads (a few minutes)"
          + ("; the model's own copy of them is freed as it goes, so it won't need twice the space" if slim else "")
          + "...", flush=True)
    repack_experts.repack(info["parts"][0], packed, slim=slim)


# ---------------------------------------------------------------- models it can download

HF = "https://huggingface.co"
CATALOG = {
    "qwen3.5-35b": {
        "about": "Qwen3.5 35B-A3B, Q5_K_M. ~5-8 words/s on 8 GB laptops, ~2 on 4 GB (normal NVMe).",
        "repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "files": ["Qwen3.5-35B-A3B-Q5_K_M.gguf"], "gb": 26.2,
    },
    "qwen3.5-122b": {
        "about": "Qwen3.5 122B-A10B, Q5_K_M. ~1 word/s on 8 GB, ~1.5-2 on 16 GB (normal NVMe; use --fast on 16 GB).",
        "repo": "unsloth/Qwen3.5-122B-A10B-GGUF",
        "files": [f"Q5_K_M/Qwen3.5-122B-A10B-Q5_K_M-0000{i}-of-00003.gguf" for i in (1, 2, 3)], "gb": 91.5,
    },
}
HELPER = {"repo": "unsloth/Qwen3.5-0.8B-GGUF", "files": ["Qwen3.5-0.8B-Q4_K_M.gguf"], "gb": 0.53}


def models_dir():
    d = Path(os.environ.get("MOE_HOME", Path.home() / "moe-models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ask(question, default_yes, assume_yes):
    if assume_yes:
        return True
    if not sys.stdin.isatty():  # nobody to answer (a script, a pipe): don't download anything
        print(f"{question} no (not interactive; use --yes)")
        return False
    try:
        a = input(f"{question} [{'Y/n' if default_yes else 'y/N'}] ").strip().lower()
    except EOFError:
        return default_yes
    return default_yes if not a else a.startswith("y")


def download(url, dst):
    """Resumable download with a progress line; writes <dst>.part and renames when complete."""
    import urllib.request
    part = Path(str(dst) + ".part")
    for attempt in range(20):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "stowaway/1.1"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = have + int(r.headers.get("Content-Length", 0))
                if r.status == 200 and have:          # server ignored the range: start over
                    have = 0
                    total = int(r.headers.get("Content-Length", 0))
                t0, last, got = time.time(), 0, 0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(8 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - last > 2:
                            last = time.time()
                            done = have + got
                            speed = got / max(time.time() - t0, 1e-3) / 1e6
                            print(f"\r  {dst.name}: {done / GB:.1f} / {total / GB:.1f} GB  ({speed:.0f} MB/s)   ",
                                  end="", flush=True)
            if part.stat().st_size >= total > 0:
                part.rename(dst)
                print(f"\r  {dst.name}: done ({total / GB:.1f} GB)" + " " * 20)
                return
        except Exception as e:  # network hiccup: resume
            print(f"\n  download interrupted ({e}); resuming...", flush=True)
            time.sleep(min(30, 2 + attempt * 3))
    sys.exit(f"could not download {url}")


def fetch(entry, into, assume_yes, what):
    """Download a catalog entry's files into `into` (asks first). Returns the first file's path."""
    paths = [into / Path(f).name for f in entry["files"]]
    missing = [(f, p) for f, p in zip(entry["files"], paths) if not p.exists()]
    if not missing:
        return paths[0]
    free = shutil.disk_usage(models_dir()).free / GB
    need = entry["gb"] * (1.1 if what == "model" else 1.0)  # packing frees the original's copy as it goes (slim)
    print(f"{what}: {entry['repo']} ({entry['gb']:.1f} GB download from huggingface.co)")
    if what == "model":
        print(f"  needs ~{need:.0f} GB of disk in {into}; {free:.0f} GB free")
    if free < need:
        sys.exit(f"not enough disk space in {into}: need ~{need:.0f} GB, have {free:.0f} GB "
                 f"(set MOE_HOME to a folder on a bigger drive)")
    if not ask(f"  download {entry['gb']:.1f} GB now?", what != "model", assume_yes):
        sys.exit("ok, not downloading")
    into.mkdir(parents=True, exist_ok=True)
    for f, p in missing:
        download(f"{HF}/{entry['repo']}/resolve/main/{f}", p)
    return paths[0]


def resolve_model(name, assume_yes, plan_only=False):
    if name in CATALOG:
        e, into = CATALOG[name], models_dir() / name
        if plan_only and not all((into / Path(f).name).exists() for f in e["files"]):
            free = shutil.disk_usage(models_dir()).free / GB
            sys.exit(f"{name} isn't downloaded yet: {e['gb']:.1f} GB download, needs ~{e['gb'] * 1.1:.0f} GB of disk "
                     f"({free:.0f} GB free in {models_dir()}).\n{e['about']}\nget it with: stowaway run {name}")
        return fetch(e, into, assume_yes, "model")
    p = Path(name)
    if p.exists():
        return p
    sys.exit(f"'{name}' is neither a file nor a known model; try 'stowaway list'")


def cmd_list():
    print("models stowaway can download and run:\n")
    for name, e in CATALOG.items():
        here = (models_dir() / name / Path(e["files"][0]).name).exists()
        print(f"  {name:14s} {e['gb']:5.1f} GB  {'(downloaded) ' if here else ''}{e['about']}")
    print(f"\nmodels are stored in {models_dir()} (set MOE_HOME to change)")
    print("any other Mixture-of-Experts GGUF file works too: stowaway run path/to/model.gguf")


def menu():
    """What you get when you double-click moe: pick a model and chat, no typing commands."""
    names = list(CATALOG)
    print("stowaway - run big AI models on an ordinary computer\n")
    for i, name in enumerate(names, 1):
        e = CATALOG[name]
        here = (models_dir() / name / Path(e["files"][0]).name).exists()
        print(f"  {i}. {name:14s} {e['gb']:5.1f} GB  {'(downloaded) ' if here else ''}{e['about']}")
    print(f"\nmodels are stored in {models_dir()}")
    try:
        pick = input(f"\nwhich model? [1-{len(names)}, Enter = 1] ").strip() or "1"
        name = names[int(pick) - 1]
    except (ValueError, IndexError, EOFError):
        sys.exit("no model picked")
    print()
    try:
        main(["run", name])
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e.code)
            input("\npress Enter to close")
            sys.exit(1)
        raise


def check_for_update():
    """One quick look at GitHub for a newer release (3 s at most, never fatal). STOWAWAY_NO_UPDATE_CHECK=1 turns it off."""
    if os.environ.get("STOWAWAY_NO_UPDATE_CHECK"):
        return
    try:
        import json
        import urllib.request
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/latest",
                                     headers={"User-Agent": f"stowaway/{VERSION}", "Accept": "application/vnd.github+json"})
        tag = json.load(urllib.request.urlopen(req, timeout=3)).get("tag_name", "")
        latest = tuple(int(x) for x in tag.lstrip("v").split(".") if x.isdigit())
        if latest > tuple(int(x) for x in VERSION.split(".")):
            print(f"update:  stowaway {tag} is available (you have v{VERSION}): https://github.com/{REPO}/releases/latest\n")
    except Exception:
        pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--version", "version", "-V"):
        print(f"stowaway {VERSION}")
        return
    if not argv and sys.stdin.isatty():
        check_for_update()
        return menu()
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv[0] == "list":
        return cmd_list()
    cmd = "run"
    if argv[0] in ("run", "plan"):
        cmd = argv.pop(0)
    if cmd == "run" and sys.stdin.isatty():
        check_for_update()
    ap = argparse.ArgumentParser(prog=f"stowaway {cmd}")
    ap.add_argument("model", help="a model name from 'stowaway list', or a path to a .gguf file")
    ap.add_argument("--cli", action="store_true", help="chat in the terminal instead of the browser")
    ap.add_argument("-p", "--prompt", help="answer one prompt and exit")
    ap.add_argument("-n", type=int, default=512, help="max tokens per answer (default 512)")
    ap.add_argument("--plan", action="store_true", help="only print the plan")
    ap.add_argument("--fast", action="store_true", help="when the model's choice of expert is close, prefer one that is "
                    "already in memory or already being read (30-40%% less reading from disk, so faster answers and a "
                    "faster first word; answers differ slightly from the full model)")
    ap.add_argument("--think", action="store_true", help="let the model think out loud before answering (better on hard "
                    "questions, but on a slow machine it can take minutes before the answer starts)")
    ap.add_argument("-y", "--yes", action="store_true", help="don't ask before downloading")
    ap.add_argument("--ram", type=float, help="pretend this many GB are free (default: measure)")
    ap.add_argument("--threads", type=int, default=min(8, max(2, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bin", help="folder with llama-server / llama-cli (default: next to moe)")
    ap.add_argument("--packed", help="existing packed experts (path without .bin/.idx)")
    ap.add_argument("--slim", action="store_true", help="when packing, free the model file's own copy of the experts "
                    "(halves disk use; the .gguf then only works with stowaway). Automatic for models stowaway downloaded")
    ap.add_argument("--experts", type=int, help="use this many experts per token instead of the model's own number "
                    "(faster on slow drives, but CHANGES the answers)")
    ap.add_argument("--draft", help="small helper model for guessing ahead ('none' to disable)")
    args = ap.parse_args(argv)
    if cmd == "plan":
        args.plan = True
    run(args)


def run(args):
    model_path = resolve_model(args.model, args.yes, args.plan)
    info = model_info(model_path)
    if not info["experts"]:
        sys.exit("this isn't a Mixture-of-Experts model; stowaway only helps with MoE models")
    fewer = args.experts and 0 < args.experts < info["k"]
    if fewer:
        info["active_expert_gb"] *= args.experts / info["k"]
    first = info["parts"][0]
    packed = Path(args.packed) if args.packed else first.parent / (first.name.split("-00001-of-")[0].removesuffix(".gguf") + "-experts-packed")
    ram = args.ram if args.ram else available_ram_gb()
    draft = find_draft(first, args.draft)
    if info["mtp"] and args.draft is None:
        draft = None  # the model's own MTP head guesses better and needs no extra RAM
    dense_packed = packed.parent / (packed.name.replace("experts-packed", "dense-packed") if "experts-packed" in packed.name
                                    else packed.name + "-dense")
    drive = drive_speed_gbps(Path(f"{packed}.bin") if Path(f"{packed}.bin").exists() else info["parts"][-1])

    print(f"model:   {first.name}  ({info['file_gb']:.1f} GB: {info['expert_gb']:.1f} GB experts, "
          f"{info['dense_gb']:.1f} GB always-needed; {info['active_expert_gb']:.2f} GB of experts per token)")
    print(f"machine: {ram:.1f} GB RAM free, {args.threads} threads, drive ~{drive:.1f} GB/s" if drive else
          f"machine: {ram:.1f} GB RAM free, {args.threads} threads")
    plan, err = make_plan(info, ram, drive)
    if err:
        sys.exit(f"can't run: {err}")
    # guessing ahead costs RAM (the helper, or MTP's extra buffers); on small machines that RAM is worth more to the
    # streamed weights (measured on a 4 GB VM: 0.6-1.3 tok/s with the helper vs 1.3-1.8 without)
    guess = bool(plan["dense_stream_gb"]) and args.draft != "none" and not plan["small"]
    if guess and not info["mtp"] and not draft and info["arch"].startswith("qwen3") and not args.plan:
        # guessing ahead roughly doubles speed here; offer the small helper model
        print("tip:     a 0.5 GB helper model makes this model ~1.5-2x faster on this machine")
        if ask("  download it?", True, args.yes):
            draft = fetch(HELPER, models_dir() / "helper", True, "helper")
    draft_gb = draft.stat().st_size / GB + DRAFT_OVERHEAD_GB if draft and guess else 0.0  # only reserve it if it's used
    if plan["dense_stream_gb"] and draft_gb:
        plan, err = make_plan(info, ram, drive, draft_gb)  # make room for the helper model
        if err:
            sys.exit(f"can't run: {err}")
    how = ("always-needed weights stay in RAM" if not plan["dense_stream_gb"] else
           f"always-needed weights streamed with a {plan['dense_stream_gb']:.1f} GB budget")
    if plan["small"]:
        print(f"small:   under {SMALL_RAM_GB:.0f} GB free, so a {plan['ctx']}-token conversation window and smaller buffers")
    print(f"plan:    {plan['cache_gb']:.1f} GB expert cache, {how}; ~{plan['read_per_token_gb']:.2f} GB read per token"
          + (f", expect roughly {plan['est_tok_s']:.1f} words/s" if plan["est_tok_s"] else ""))
    if guess and info["mtp"] and not draft:
        print("guessing: the model's built-in MTP head guesses ahead, the big model checks (faster; rarely a word differs)")
    elif guess and draft:
        print(f"guessing: {draft.name} guesses ahead, the big model checks (faster; rarely a word differs)")
    elif draft and not plan["dense_stream_gb"]:
        print(f"note:    not using {draft.name}: guessing only helps when the always-needed weights don't fit in RAM")
    if fewer:
        print(f"quality: using {args.experts} of {info['k']} experts per token - faster, but answers will differ "
              f"from the full model")
    # --fast: batch-aware routing while reading the question always cuts reads. Cache-aware routing while answering only
    # pays when the cache holds at least one token's worth of experts: measured -39% reads on the 35B at 8 GB (cache =
    # 3.6 tokens' worth), nothing on the 122B Q8 at 8 GB (0.13), where it would only change the output.
    fast_answer = args.fast and plan["cache_gb"] >= info["active_expert_gb"]
    if args.fast:
        print("fast:    while reading your question, words whose choice of expert is close share experts that are read "
              "anyway (a faster first word)" + ("; while answering, prefers experts already in memory" if fast_answer else
              "; answering uses the model's own choices (the memory cache is too small here for that to help)")
              + ". Answers differ slightly from the full model")
    if args.plan:
        return

    slim = args.slim or models_dir() in first.resolve().parents
    ensure_packed(info, packed, slim)
    env = dict(os.environ,
               MOE_CACHE_GB=f"{plan['cache_gb']:.2f}", MOE_IO_THREADS="4", MOE_PREGATE=str(plan["pregate"]),
               EXPERT_CACHE_PACKED=str(packed), EXPERT_CACHE_CHUNK_KB="8192", LLAMA_NO_MMAP_PREFETCH="1",
               CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
    if args.fast:
        env["MOE_BATCH_BONUS"] = "1.0"  # batch-aware routing while reading the question: -31% reads, KLD 0.016 (RESULTS.md 20)
        if fast_answer:
            env["MOE_CACHE_BONUS"] = "1.0"  # cache-aware routing while answering: -39% expert reads, KLD 0.029 (RESULTS.md 16)
    if plan["dense_stream_gb"]:
        ensure_dense_packed(info, dense_packed)
        env["EXPERT_CACHE_DENSE_GB"] = f"{plan['dense_stream_gb']:.2f}"
        env["EXPERT_CACHE_DENSE_PACKED"] = str(dense_packed)
    common = ["-m", str(first), "-ngl", "0", "--no-repack", "--no-op-offload", "-c", str(plan["ctx"]),
              "-b", str(plan["batch"]), "-ub", str(plan["batch"]),
              "-t", str(args.threads), "-tb", str(os.cpu_count() or args.threads), "-n", str(args.n),
              "--no-warmup",  # warmup runs every expert once, which fills the small cache with junk
              "-rea", "on" if args.think else "off"]
    if guess and info["mtp"] and not draft:
        common += MTP_FLAGS
    elif guess and draft:
        common += ["-md", str(draft)] + SPEC_FLAGS
    if fewer:
        common += ["--override-kv", f"{info['arch']}.expert_used_count=int:{args.experts}"]
        env["MOE_EXPERTS_USED"] = str(args.experts)

    if args.prompt or args.cli:
        cmd = [str(find_bin("llama-cli", args.bin))] + common
        if args.prompt:
            cmd += ["-p", args.prompt, "-st", "--simple-io", "--no-display-prompt"]
        sys.exit(subprocess.call(cmd, env=env))

    port = free_port(args.port)
    cmd = [str(find_bin("llama-server", args.bin))] + common + ["--host", "127.0.0.1", "--port", str(port)]
    log_path = models_dir() / "logs" / f"stowaway-server-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{port}"
    print(f"loading the model (the engine's messages go to {log_path})...", flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    import urllib.request
    t0 = time.time()
    while True:
        try:
            if urllib.request.urlopen(f"{url}/health", timeout=1).status == 200:
                break
        except Exception:
            if proc.poll() is not None:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
                print("\n".join(tail))
                sys.exit(f"\nthe engine stopped before the chat was ready; the full log is in {log_path}")
            if time.time() - t0 > 900:
                proc.terminate()
                sys.exit(f"the chat didn't start within 15 minutes; see {log_path}")
            time.sleep(1)
    print(f"\nYour chat is ready: {url}")
    print("It should open in your web browser now. If it doesn't, copy that address into your browser.")
    print("Keep this window open while you chat. Closing it (or pressing Ctrl+C) turns the AI off.\n", flush=True)
    webbrowser.open(url)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("stopped.")


if __name__ == "__main__":
    main()
