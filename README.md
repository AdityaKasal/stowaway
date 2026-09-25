# stowaway

Run big AI models on an ordinary computer: no graphics card, 8 GB of RAM.

stowaway runs **Qwen3.5-122B-A10B at Q5** (a 91.5 GB model) on a machine with 8 GB of RAM and no GPU, and
**Qwen3.5-35B-A3B at Q5** (26 GB) at conversation speed on the same machine. Normally both need a big GPU or a lot
of RAM. The quality is unchanged: it's the full Q5 model, not a smaller or more compressed one.

![moe chatting with Qwen3.5-35B-A3B Q5 in the browser](docs/chat.gif)

*Real time, not sped up: Qwen3.5-35B-A3B Q5 in the browser chat that `moe` opens. Recorded on a MacBook Pro (M5 Pro)
with `moe` limited to 5.5 GB of RAM, about what an 8 GB laptop has free. That Mac's SSD is fast; a typical 8 GB
laptop gets about half this speed (see the table below).*

## Download

Get the zip for your computer from [Releases](../../releases/latest), unzip it, and double-click `stowaway`.
Pick a model; it shows the download size and asks before downloading anything. After a one-time setup, a chat opens
in your browser.

| | 8 GB RAM, no GPU, typical laptop SSD (3 GB/s) |
|---|---|
| Qwen3.5-35B-A3B Q5 (26 GB download) | ~5-8 words per second (~2 on a 4 GB machine) |
| Qwen3.5-122B-A10B Q5 (92 GB download) | ~1 word per second (~1.5-2 with 16 GB of RAM and `--fast`) |

You need an SSD and about the model's size in free disk space, plus 10%. From a terminal:

```
stowaway list                      models it can download
stowaway run qwen3.5-35b           download (asks first), set up, chat in the browser
stowaway run qwen3.5-35b --cli     chat in the terminal
stowaway run qwen3.5-35b --fast    faster answers and a faster first word; answers differ slightly (see below)
stowaway run qwen3.5-35b --think   let the model think before answering (off by default: slow on slow machines)
stowaway plan qwen3.5-122b         show the memory plan and expected speed
stowaway run path/to/model.gguf    any other Mixture-of-Experts GGUF model (add --slim to halve its disk use)
```

For a step-by-step walkthrough for non-technical people, see the setup guide (ask Aditya for the link).

At start it checks GitHub for a newer version (set `STOWAWAY_NO_UPDATE_CHECK=1` to turn that off); `stowaway --version`
shows yours.

Windows says "Windows protected your PC" the first time: click More info, then Run anyway. On a Mac, if it
won't open, run `xattr -dr com.apple.quarantine .` in the unzipped folder. Linux needs glibc 2.34+ and an
x86-64 CPU with AVX2.

## How it works

These are Mixture-of-Experts models. The 122B has 256 "experts" per layer, and each word uses only 8 of them, so
each word needs about 2.6 GB of the 85 GB of experts. `moe` keeps the experts on the SSD and reads only the ones the
model picks. It keeps recently used experts in a fixed RAM cache and uses the next layer's router to guess and
preload experts before they're needed. If even the always-needed weights don't fit in RAM, they are streamed too,
one layer at a time. When that happens, a small helper model (or the model's own built-in predictor) guesses a few
words ahead so each pass over the weights produces several words.

Speed comes down to drive speed divided by the bytes read per word. All the measurements, including the ideas that
didn't work, are in [RESULTS.md](RESULTS.md).

Setup packs the experts once into a file laid out for fast reads. For models stowaway downloads, the model file's
own copy of the experts is freed as it goes (the file keeps its size but the space is released), so a model needs
about its own size on disk instead of twice.

`--fast` turns on cache-aware routing. When the expert the model wants isn't in memory but a nearly-as-good one is,
it uses that one (39% less reading while answering, on the 8 GB setup). While reading your question, it prefers
experts that other words of the question already need, which makes the first word come ~75% sooner for the 122B on
8 GB. The quality cost is measured and small (RESULTS.md sections 16 and 20), but it's off by default because the
answers change slightly.

The cache never changes the output: it's bit-identical to plain llama.cpp. Guessing ahead is the same model at the
same quality, but llama.cpp's batched check rounds slightly differently, so an occasional word can differ.

## Related work

Streaming Mixture-of-Experts weights from storage is not a new idea, and this project builds on others' work:

- [colibri](https://github.com/JustVugg/colibri): a pure-C engine that streams experts from NVMe to run
  744B-2.8T models on machines from ~25 GB of RAM up, with a hardware planner and MTP speculation. It targets the
  biggest models; stowaway targets the smallest machines (8 GB, no GPU) with standard GGUF files at Q5-Q8.
- [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) (stream MoE experts from disk) and
  [Hypura](https://github.com/ggml-org/llama.cpp/discussions/20852) (GPU/RAM/NVMe placement on Macs).
- Research: *LLM in a flash* (Apple, 2023: running models larger than DRAM from flash, bundling weights for bigger
  reads), *Fast Inference of Mixture-of-Experts Language Models with Offloading* (Eliseev & Mazur, 2023: LRU expert
  cache, speculative expert loading), *Pre-gated MoE* (ISCA 2024: predicting the next layer's experts), and work on
  cache-aware expert selection.

What stowaway adds: an 8 GB target with measured results, streaming the always-needed weights too when they don't
fit, opt-in cache-aware routing with measured quality cost, slim packing, and a one-click app.

## Build from source

```bash
./setup-llama.sh            # clones llama.cpp at d2e5458 and applies patches/moe-stream.patch
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build llama.cpp/build -j --target llama-cli llama-server
pip install numpy pyyaml
python moe.py list          # the app is moe.py; release builds name it stowaway
```

The release builds use `pc/build-dist.ps1` (Windows), `vm/build-dist.sh` (Linux, Ubuntu 22.04) and the Mac recipe
further down. Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).

# Research notes

## What's here

| Path | What it is |
|---|---|
| `moe.py` | The `moe` program (`moe_run.py` is an older name for `moe.py run`) |
| `patches/moe-stream.patch` | Our changes to llama.cpp commit `d2e5458` (`setup-llama.sh` applies them): hooks in ggml-cpu for expert and dense weights, `LLAMA_NO_MMAP_PREFETCH`, and `common/moe-stream.h` (the cache, dense streaming and prediction), started from `common_init_from_params` when `MOE_CACHE_GB` is set |
| `expert-logger/expert-logger.cpp` | Runs prompts through the model. Records which experts the router picks (`experts.csv`), and optionally prefetches (`--prefetch`) or uses our own expert cache (`--expert-cache-gb N`) |
| `pc/ram_limit.py` | Makes a PC behave like one with less RAM (locks memory away; no settings changed) |
| `analyze.py` | Turns `experts.csv` into reuse / cache hit rate / prediction numbers (`data/summary.json`) |
| `policies.py` | Replays a router trace through LRU, SLRU, 2Q, ARC, LFU-decay and the optimal policy |
| `repack_experts.py` | Writes a copy of the experts with each expert's pieces side by side (packed mode) |
| `pack_dense.py` | Writes the always-needed weights layer by layer, for one-read-per-layer streaming |
| `stripe_experts.py` | Copies part of the packed experts to a second drive (whole experts, or every expert's tail) |
| `bench.sh` | Mac benchmark modes (`cpu`, `cpu-prefetch`, `stream`, `prefetch`, `naive`) |
| `pc/*.ps1` | The same for the Windows PC (copies live in `C:\Users\FSociety\moe-router-study\scripts`) |
| `prompts.tsv`, `bench-prompts.tsv` | 24 study prompts across 6 topics; 4 quick benchmark prompts |

### expert-logger options (on top of normal llama.cpp flags)

| Option | What it does |
|---|---|
| `--expert-cache-gb N` | Our expert cache: N GB of RAM for experts, least-recently-used eviction, whole-expert parallel reads that bypass the OS file cache. Use with `--no-op-offload` |
| `--io-threads N` | Parallel reads for the cache (default 16; 32 was slower on the PC) |
| `EXPERT_CACHE_CHUNK_KB` (env) | Read size (default 4096 = one whole expert; smaller was much slower on the PC) |
| `--pregate M` | Predict the next layer's experts with its router and preload the top M. Best: `--pregate 6` (~5% faster on the PC 122B, 3x less waiting on the Mac). Works with `--no-hook` |
| `--pregate-depth D` | Guess D layers ahead (2 was slower) |
| `PREGATE_PROMPT_TOKENS`, `PREGATE_PROMPT_THREADS` (env) | Prompt look-ahead (on with `--pregate`): predict from up to 64 sampled prompt tokens on 8 threads. `PREGATE_NO_PROMPT=1` turns it off |
| `-tb N` | llama.cpp's thread count for prompt processing; `-tb 20` on the PC made prompts ~25% faster |
| `--prefetch` | Older approach: hint the OS to read the chosen experts (no cache of our own) |
| `--no-log` / `--no-hook` | Skip writing experts.csv / skip watching the router entirely (fastest) |
| `EXPERT_CACHE_PACKED=<path>` (env) | Read experts from the packed copy made by `repack_experts.py` (one ~6.9 MB block per expert instead of three pieces). Output is identical; ~2x faster on the PC |
| `EXPERT_CACHE_STRIPE`, `EXPERT_CACHE_TAIL` (env) | Second-drive copies made by `stripe_experts.py` (`3 1` = every third whole expert; `tail 0.34` = last 34% of every expert). Use both: tail split while generating, whole experts while reading prompts. ~17% faster on the PC |
| `EXPERT_CACHE_SPEC_LIMIT`, `EXPERT_CACHE_SPEC_AGE` (env) | Pre-gating tuning (I/O threads guesses may use; how long an expert must be unused before a guess may evict it). Defaults are best |
| `EXPERT_CACHE_NO_LOCK=1` (env) | Windows: don't lock the cache in RAM |
| `LLAMA_NO_MMAP_PREFETCH=1` (env) | Don't ask the OS to read the whole model at load (matters when the model is much bigger than RAM) |

### Mac

```bash
./bench.sh cpu-prefetch                                       # stream experts, macOS page cache + our prefetch hints
./bench.sh cpu --expert-cache-gb 4 --no-op-offload --no-hook  # our own cache, 4 GB of RAM for experts (~20 tok/s)
.venv/bin/python analyze.py data models/Qwen3.5-35B-A3B-Q5_K_M.gguf
```

Flags that matter: `-ngl 0` (Metal maps the whole file otherwise and runs out of memory), `--no-repack`
(otherwise llama.cpp copies the whole model into RAM), `-b 512`.

### PC (from the Mac: `ssh fsociety`)

```powershell
cd C:\Users\FSociety\moe-router-study
powershell -ExecutionPolicy Bypass -File scripts\run-logger.ps1 <run-name> <model.gguf> bench-prompts.tsv --no-log -ngl 99 --n-cpu-moe 24 -n 128

# 122B (91.5 GB), best settings found (~7.6 tok/s, long prompt ~7.7 s). One-time setup:
#   .\.venv\Scripts\python repack_experts.py <model-00001-of-00003.gguf> models\122b\experts-packed
#   .\.venv\Scripts\python stripe_experts.py models\122b\experts-packed E:\moe\experts-stripe 3 1
#   .\.venv\Scripts\python stripe_experts.py models\122b\experts-packed E:\moe\experts-tail tail 0.34
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$env:EXPERT_CACHE_PACKED = "models\122b\experts-packed"
$env:EXPERT_CACHE_STRIPE = "E:\moe\experts-stripe"
$env:EXPERT_CACHE_TAIL   = "E:\moe\experts-tail"
$env:EXPERT_CACHE_CHUNK_KB = "8192"
powershell -ExecutionPolicy Bypass -File scripts\run-logger.ps1 122b models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf bench-prompts.tsv --no-hook -ngl 99 --n-cpu-moe 44 --no-op-offload --expert-cache-gb 18 --io-threads 6 --pregate 6 -t 10 -tb 20 -n 128
```

Rebuild after changing code: `powershell -ExecutionPolicy Bypass -File scripts\build-errors.ps1`.
Long jobs run as scheduled tasks via `scripts\run-task.ps1` so they survive SSH disconnects.

### If you ever want to send the patch to llama.cpp

Their rules (`llama.cpp/AGENTS.md`) require that you understand and can defend every line yourself, and they
ban AI-written PR descriptions and automated submissions. Open an issue to discuss the idea first.

### Ready-made downloads (no Python needed)

`dist/zips/moe-{mac,windows,linux}.zip` each hold `moe` (the launcher, from `moe.py` via PyInstaller), plus standalone
CPU-only `llama-cli` and `llama-server` and a README.txt. Double-click `moe` for a menu, or run `moe list`,
`moe run qwen3.5-35b`, or `moe plan <model>`. Models go to `~/moe-models` (`MOE_HOME` changes this).
`moe_run.py` still works as an alias for `moe.py run`.

How they are built:
- Mac (arm64, macOS 13+): `cmake -S llama.cpp -B build-dist -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_METAL=OFF
  -DGGML_BLAS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0`, then PyInstaller
  `--onefile --paths llama.cpp/gguf-py moe.py`.
- Windows x64: `pc/build-dist.ps1` (MSVC, static CRT, AVX2, no OpenMP/CUDA). The .exe files depend only on system DLLs.
- Linux x64: `vm/build-dist.sh` in WSL Ubuntu 22.04 (static libstdc++, needs glibc 2.34+, AVX2).

Tested from a clean folder on each: Mac 35B 21-23 tok/s; Windows (PC, CPU only) 10-11 tok/s; Linux on Ubuntu 22.04 and
26.04 (model read through WSL's /mnt/c, so slow). The browser chat page is embedded in llama-server on all three.
