# Results (2026-09-23)

All speeds are generation speed (tokens/s) on the 4 bench prompts unless noted. Raw logs: `results/` (Mac) and
`results/pc/` (copied from the PC).

## Machines and models

| | Mac | PC |
|---|---|---|
| Hardware | M5 Pro, 24 GB unified memory, internal SSD | Core Ultra 7 265K, 32 GB DDR5-4800, RTX 5060 Ti 16 GB, Samsung 990 EVO Plus |
| Model | Qwen3.5-35B-A3B Q5_K_M: 26.2 GB (23.6 GB experts + 2.7 GB always-needed) | Qwen3.5-122B-A10B Q5_K_M: 91.5 GB (84.9 GB experts + 6.6 GB always-needed) |
| Expert data per token | 737 MB | 2,652 MB |

## 1. Getting the 35B Q5 to run on the 24 GB Mac

| Setup | tok/s | Notes |
|---|---|---|
| All on GPU (Metal) | crash | Metal maps nearly the whole file into one buffer; out of memory |
| CPU, default | 3-5 | Repacking copied the whole model into a 24 GB RAM buffer |
| CPU, `--no-repack` (stream from file) | 12-20 | Process owns 0.3 GB; macOS page cache holds ~17 GB of the file |
| + `--prefetch` (F_RDADVISE hints) | 18-23 | |
| Our expert cache, 4 GB | 20-22 | |
| Our expert cache, 12 GB | 22-28 | |

Caveat: on the Mac, part of what our cache "read" was served from macOS's own file cache, which still held the model
from earlier runs. Making it clean needs `sudo purge` before each run.

## 2. Router behavior (does the idea hold?)

| | 35B (5,986 tokens) | 122B (3,060 tokens) | Random routing |
|---|---|---|---|
| Experts reused from the previous token | 35.4% | 33.2% | 3.1% |
| ...from the last 4 tokens | 55.3% | 50.8% | 11.9% |
| LRU cache hit rate, 30% of experts cached | 81.4% | 75.0% | 30% |
| LRU, 60% cached | 94.4% | 90.4% | 60% |
| Optimal (Belady), 30% cached | 91.1% | 87.7% | |
| Next-layer prediction from co-occurrence (16 guesses) | 46.6% | 44.5% | 6.2% |
| Next-layer prediction by pre-gating (12 guesses, measured live) | 89.6% | 92.9% | |
| Pre-gating, 8 guesses | 79.0% | 81.8% | |

The patterns barely change between 35B and 122B. LRU beats LFU at every size.

## 3. 122B (91.5 GB) on the PC (32 GB RAM + 16 GB VRAM)

Settings for all runs: `-ngl 99 --n-cpu-moe 44 --no-op-offload -t 10`, `LLAMA_NO_MMAP_PREFETCH=1`.

| Setup | tok/s | SSD read rate | Notes |
|---|---|---|---|
| Windows pages the file in (no cache) | 1.6-2.2 | ~1 GB/s | quiet PC |
| **Our cache, 19 GB (22% of experts), 4 MB reads** | **3.4-4.0** | | quiet PC; best result |
| Our cache, 19 GB, + pre-gating v1 (8 guesses) | 2.6-3.6 | | v1, superseded; see section 5 for v2 |
| Our cache, 19 GB, 1 MB reads, 20 threads | 1.9-2.2 | | measured while Defender and Tixati were busy |
| Our cache, 9 GB | 1.5-1.6 | | (busy PC) |
| 24-prompt router study (logging on, 128 tokens each) | 2.3-3.0 | ~2 GB/s | (busy PC) |

Raw SSD ceiling (`pc/ssd_test.py`, unbuffered 4 MB reads): 6.2 GB/s with 4-16 threads, sequential or random.
The cache averages 2-2.5 GB/s while the model waits, because each layer only needs ~19 MB and the next layer's
experts aren't known until it gets there. Closing that gap is the next step.

Tuning findings: 16 I/O threads and whole-expert 4 MB reads are best. More threads or smaller reads were much
slower (32 threads / 256 KB: 1.0-1.4 tok/s). Limiting llama.cpp to 10 compute threads helped a little: the other
threads spin while waiting on reads.

## 4. 35B on the PC (fits in RAM + VRAM, no streaming needed)

| Setup | tok/s |
|---|---|
| `--n-cpu-moe 24` (best split) | 46 |
| all experts in RAM (`--n-cpu-moe 40`) | 32 |
| CPU only | 14 |
| "everything on GPU" (overflows into shared memory) | 6 |

## 5. Making pre-gating pay off (2026-09-23, 23:00-23:35)

Version 1 read the hidden state through llama.cpp's eval callback. That paused the GPU twice per layer (96 times per
token) and broke CUDA graphs: about +43 ms per token, which wiped out any gain. Version 2 predicts from the MoE input
inside our own ggml hook (the data is already on the CPU, so no extra pauses). It also has a separate low-priority queue
for guesses (at most 8 I/O threads), cancels or demotes wrong guesses as soon as their layer runs, lets guesses evict
only experts unused for ~3 tokens, and promotes guesses to urgent when the model asks for them.

PC, 122B, 19 GB cache, `--no-hook`, same session, Tixati closed, baseline run first and last:

| Setup | median ms/token | tok/s | needed experts still loading |
|---|---|---|---|
| No pre-gating (first / last run) | 329 / 328 | 2.7-3.4 | 26.0% |
| **v2, 6 guesses, 1 layer ahead** | **312** | **2.7-3.7** | 21.5% |
| v2, 8 guesses, 1 layer ahead | 341 | 2.6-3.2 | 22.7% |
| v2, 8 guesses, 2 layers ahead | 342 | 2.5-3.2 | 21.1% |
| v1 (callback), 8 guesses | 364 | 2.4-3.0 | |
| v1 (callback), 12 guesses | 438 | 2.0-2.5 | |

Mac, 35B, 4 GB cache, `--no-hook`: no pre-gating 18.1-20.1 tok/s (13.2 s waiting on reads) vs v2 with 8 guesses
20.0-21.1 tok/s (4.3 s waiting).

Takeaways: fewer, more confident guesses win. Guessing 2 layers ahead loses accuracy faster than the extra head
start helps. On the PC about 21% of needed experts are still mid-read when asked for, even when guessed right. The
remaining limit is how fast a single layer's reads finish (average read rate while waiting is ~1.8 GB/s against
6.2 GB/s raw), not prediction.

## 6. Finding the real bottleneck, and the packed layout (2026-09-24, 00:00-01:30)

Measured instead of guessed (timing counters in the cache, `pc/read_latency.py`, Windows perf counters):

- **Not memory paging and not page locking.** A 13 GB cache left 2 GB free and reads were just as slow; locking the
  cache in RAM (`VirtualLock`, now on by default on Windows) didn't change read times either.
- **The SSD's real ceiling for expert-sized reads:** one 2.3 MB read takes 0.75 ms alone, but the bandwidth (~4.2-4.5
  GB/s) is shared by everything in flight, so 16 at once take ~8.6 ms each. Our urgent reads were sharing with
  background reads.
- **Generation was bandwidth-bound:** ~0.8 GB read per token at ~4.5 GB/s = ~180 ms of a ~300 ms token.
- **Smarter eviction doesn't help:** simulated on the 122B trace (`policies.py`), SLRU, 2Q, ARC and LFU-with-decay
  were all within +/-0.5 points of LRU (65% at 19% cache). Only a perfect future-knowing policy is better (83%).
- **Bigger reads do:** 6.8 MB reads reach 5.8-6.0 GB/s vs 4.2 for 2.3 MB. The GGUF stores an expert as three pieces
  far apart, so `repack_experts.py` writes a second copy of the experts (85 GB, 4 minutes, spot-checked byte for byte)
  with each expert's gate/up/down side by side. The cache reads one 6.9 MB block per expert
  (`EXPERT_CACHE_PACKED=models\122b\experts-packed`).

Final back-to-back run on the quiet PC, 122B, 4 prompts x 128 tokens, `-t 10 --no-hook`:

| Setup | median tok/s | range | tokens identical to no-cache run |
|---|---|---|---|
| Windows pages the file in (no cache) | 0.6 | 0.5-0.7 | (reference) |
| Cache v1 (yesterday: 19 GB, 16 threads, 4 MB reads) | 2.9 | 2.4-3.0 | 511/511 |
| **Packed + pre-gating 6, 18 GB cache, 4 I/O threads, 8 MB reads** | **5.9** | **4.5-6.1** | **511/511** |
| Same with 20 GB cache | 6.3 | 5.2-6.8 | 511/511 |

The 20 GB cache leaves Windows ~550 MB free and pushes other programs to disk, so 18 GB is the everyday setting.
The no-cache speed varies a lot between runs (0.6 here, 1.6-2.2 with 64-token answers earlier).

Tuning that led here (64-token runs): packed alone took median 300 -> 187 ms per token; pre-gating 6 on top ->
164 ms (12% more, versus ~5% without packing, because one guess now brings a whole expert); 8 guesses or 2 layers
ahead were worse; 2-4 I/O threads are enough.

Mac 35B: packing didn't help (15-16 vs 16-18 tok/s). The Mac barely waits on the SSD (<1 ms per wait); it's limited
by CPU speed. The comparison also favors the unpacked file, which macOS still partly caches.

## 7. Faster prompt reading (2026-09-24, 01:40-02:10)

Before a model can answer, it processes the whole prompt at once. Even a 20-token prompt needs ~60% of every
layer's experts (~20 GB of reads on the 122B); a 366-token prompt needs ~77% (~69 GB). It was ~85% waiting on the SSD.

What helped:
1. **`-tb 20`**: use all 20 cores for prompt math (generation still uses `-t 10`).
2. **Prompt look-ahead** (automatic with `--pregate`): at each layer, run the next layer's router on the prompt's
   tokens (a sample of up to 64, spread over 8 threads) and preload the union of their experts while the current
   layer computes. Routing every token on one thread cost ~6 s for a 366-token prompt; spreading and sampling made it
   nearly free.

PC, 122B, packed, 18 GB cache (same seed, outputs identical in every pair):

| Prompt | Before (this morning) | + `-tb 20` | + prompt look-ahead |
|---|---|---|---|
| 4 short prompts (18-27 tokens), total | 18.6 s | 13.9 s | **12.2-12.7 s** |
| 1 long prompt (366 tokens) | 14.6 s | 12.2 s | **9.1 s** |

That's the SSD's rated limit: ~69 GB in ~9 s is ~7 GB/s. Loading the whole next layer instead of predicted experts
(`PREGATE_PROMPT_ALL`) was slower (10.0 s); sampling 32, 64 or all tokens made no real difference (9.1-9.3 s).

Mac, 35B, 4 GB cache: short prompts 4.8 -> 4.0 s total, long prompt 6.1 -> 5.8 s.

## 8. Using the PC's second SSD (2026-09-24, 03:00-04:00)

The PC has a second NVMe drive: E:, a WD SN730 (PCIe 3). With 6.8 MB reads it does ~3.15 GB/s vs ~6.66 GB/s for the
990 EVO Plus, so it should carry about a third of the reads. `stripe_experts.py` copies part of the packed expert
file there (the full file stays on C:). There are two layouts, and the cache uses both:

- **Whole experts** (`EXPERT_CACHE_STRIPE`, 28.2 GB): every third expert lives on E:. It helps prompts, which load
  many experts at once. It doesn't help generation, because a single expert still comes from one drive.
- **Tail split** (`EXPERT_CACHE_TAIL`, 28.9 GB): the last 34% of every expert lives on E:. Each expert becomes two
  parallel reads (head from C:, tail from E:), so even one expert loads faster. It helps generation.
- **Both at once**: split reads while generating, whole-expert reads while reading a prompt.

PC, 122B, 18 GB cache, pre-gating 6, `-t 10 -tb 20`. Separate batches (outputs identical, 511/511 tokens):

| Layout | Generating | 4 short prompts | Long prompt |
|---|---|---|---|
| One drive | 5.9 tok/s | 17.0 s | 11.4 s |
| Whole experts on E: | 5.9 tok/s | 12.7 s | 8.3 s |
| Tail split | 6.6 tok/s | 13.6 s | 9.8 s |

Confirmed back to back (one drive, both drives, one drive again):

| | Generating | 4 short prompts | Long prompt |
|---|---|---|---|
| One drive (before / after) | 6.5 / 6.5 tok/s | 12.5 / 12.5 s | 9.4 / 9.3 s |
| **Both drives, both layouts** | **7.6 tok/s (+17%)** | **10.3 s (-18%)** | **7.7 s (-18%)** |

Note the one-drive speed itself varies between batches (5.9-6.5 tok/s), so compare within a batch.

**Speculative decoding with Qwen's MTP head: not pursued.** llama.cpp supports it (`--spec-type draft-mtp`), but it
can't export Qwen3.5's MTP head as a small add-on file, so it would mean a different 94 GB download. More
importantly, checking k draft tokens means loading every expert those tokens use, and neighboring tokens share only
~35% of their experts. Three tokens cost ~2.3x the reads for ~2.2 tokens of progress, roughly break-even when reading
is the bottleneck.

## 9. The real target: a cheap machine (8 GB RAM, no GPU)

The goal is running big models on hardware anyone has, not a gaming PC. Simulated on the PC with no GPU
(`CUDA_VISIBLE_DEVICES=-1`), 4 CPU threads, experts read only from the PCIe 3 drive (E:, ~3.1 GB/s), and slower
drives simulated with a read-speed cap (`EXPERT_CACHE_MAX_MBPS`). Memory is the program's measured peak working set.
Model: Qwen3.5-35B-A3B Q5_K_M (26.2 GB), packed (`pc/budget.ps1`).

| Expert cache | Program's peak RAM | Generation | Short prompt | Long prompt (366 tok) |
|---|---|---|---|---|
| 1 GB | 3.4 GB | 4.0 tok/s | 2-3 s | |
| **2 GB** | **4.4 GB** (4.4 with `-b 128`, 0.4 GB less private) | **5.8 tok/s** | **2-3 s** | **9.8 s** |
| 3 GB | 5.3 GB | 6.5 tok/s | 2-3 s | |

Slower drives (2 GB cache):

| Drive | Generation | Short prompt |
|---|---|---|
| PCIe 3 NVMe, ~3.1 GB/s (real) | 5.8 tok/s | 2-3 s |
| Slow NVMe, 1.5 GB/s (simulated) | 1.4 tok/s | ~10 s |
| SATA SSD, 0.5 GB/s (simulated) | 0.9-1.0 tok/s | 13-16 s |

Pre-gating neither helps nor hurts on SATA (0.9 vs 1.0 tok/s). It's pure bandwidth: ~0.38 GB per token from disk at
0.5 GB/s. The simulated 1.5 GB/s result is probably pessimistic: the cap serializes reads, where a real drive
overlaps them.

Caveats: the PC's CPU cores are faster than a budget laptop's, even limited to 4 threads. Windows may also keep
parts of the always-needed weights in its file cache outside the process's working set.

What this means:
- **A 26 GB Q5 model runs in ~4.4 GB of RAM with no GPU** at ~6 tok/s on any NVMe laptop. An 8 GB laptop has room.
- **The drive matters more than RAM or GPU.** SATA SSDs cap it around 1 tok/s. NVMe (in most laptops since ~2019)
  is the practical minimum.
- **Model choice matters more than model size.** What has to fit is the always-needed part (2.7 GB for the 35B,
  6.6 GB for the 122B) and the expert data per token (0.74 GB vs 2.65 GB). The 122B needs ~10+ GB of RAM and would
  be ~3.5x slower per token, so it isn't an 8 GB-laptop model. MoE models with few active parameters are.

## 10. The 122B on an 8 GB machine, and the one-command tool (2026-09-24 afternoon)

**Honest simulation of an 8 GB laptop:** `pc/ram_limit.py` locks away memory until only ~5.5 GB is available (it
re-locks anything Windows frees later, and changes no settings). No GPU, 4 threads.

- The 35B holds up under the real limit: 5.9 tok/s with a 2 GB expert cache.
- The 122B's always-needed weights (5.8 GB) don't fit. Leaving them to Windows was chaotic: the same setting gave
  0.2-1.2 tok/s from run to run (Windows evicts them in the worst order, since they're used in a fixed cycle).
  OS-level load-ahead and pinning helped sometimes but weren't repeatable.
- **Fix: dense streaming** (`EXPERT_CACHE_DENSE_GB`). A second hook in ggml-cpu (`ggml_cpu_set_weight_hook`) lets
  us hand ordinary matmuls their weights from our own memory. The output layer and as many whole layers as fit are
  locked in RAM once; the rest cycle through a 4-slot ring, read a few layers ahead with big direct reads. Memory use
  is fixed. (A ring-slot bug corrupted output when the streamed layer count wasn't a multiple of 4; found by the
  token check and fixed by assigning slots by step count.)

122B Q5, ~5 GB available, no GPU, 4 threads, one prompt x 32 tokens, all outputs identical to the normal run:

| Setup | Speed | Program's RAM |
|---|---|---|
| Windows handles the always-needed weights | 0.2-1.2 tok/s (not repeatable) | |
| **Dense streaming 3.2 GB + expert cache 0.5 GB, prediction off** | **0.9 tok/s (1.12 s per token), repeatable** | ~4.8 GB |
| Dense streaming 3.0 GB + expert cache 0.5 GB | 0.9 tok/s | ~4.8 GB |
| Dense streaming 2.0 GB + expert cache 1 GB | 0.8 tok/s | ~4.4 GB |

That's close to what the drive allows: each token needs ~3.3 GB of streamed dense weights plus ~2.65 GB of experts,
~6 GB, i.e. ~0.9 s at this drive's ~6.6 GB/s. A typical PCIe 3 laptop drive (~3 GB/s) would give ~0.45 tok/s.

**One-command tool: `moe_run.py`.** It checks free RAM, CPU and drive speed, reads the model's layout, packs the
experts once, picks the memory split (always-needed weights in RAM if they fit, dense streaming if not), prints an
estimate, and starts llama.cpp's own chat (browser via `llama-server`, or `--cli`, or `-p` for one answer). The
cache now lives in `llama.cpp/common/moe-stream.h` and turns on in any llama.cpp program through environment
variables (`MOE_CACHE_GB` etc.), hooked in `common_init_from_params`.

Tested end to end:
- Mac, 35B, pretending 5.5 GB free: 11.3 tok/s. Pretending 3.8 GB free (dense streaming): 5.6 tok/s.
- PC, 122B, under the ~5 GB limit: 0.8 tok/s through `llama-cli`. The tool's estimate (0.9) matches the measured runs.
- PC and Mac `llama-server` serve the built-in chat page.

Weak spot: with tiny caches, reading a prompt is slow (~1-3 tokens/s), because even a short prompt touches over
half of all experts.

## 11. Making the 122B faster on 8 GB (2026-09-24, 15:20-16:00)

Same honest 8 GB simulation (~5.5 GB available, no GPU, 4 threads), measured through `llama-cli` (what users run),
greedy decoding so every variant must print identical text. It did, in every run below (on this Windows build;
see section 14: speculative decoding is not guaranteed bit-identical in general).

| Step | 122B speed | Why it helps |
|---|---|---|
| Start (dense streaming 3.2 GB + 0.5 GB expert cache) | 0.6 tok/s | |
| + **packed dense weights** (`pack_dense.py`, `EXPERT_CACHE_DENSE_PACKED`) | 0.9 | one big read per streamed layer instead of ~12 scattered ones |
| + **`--no-warmup`** | 1.1 | llama.cpp's warmup runs every expert once and fills the small cache with junk |
| + **helper model** (Qwen3.5-0.8B Q4, 0.5 GB), 3 guesses | 1.7 | the 122B checks several guessed tokens in one pass, reading the streamed weights once |
| + guess up to 12, stop below 80% confidence | **2.3** | fewer wasted guesses, longer runs when the helper is sure |

The helper's gain depends on the text (all identical output):

| Prompt | Without helper | With helper |
|---|---|---|
| Explanation ("why is the sky blue") | 1.1 | 2.3 |
| Story opening | 0.9 | 2.0 |
| Python function | 0.8 | 1.2 |

Other findings:
- More locked dense weights help a little: 3.2 -> 3.8 GB was 1,122 -> 1,030 ms per token (the budget is tight).
- 4 or 6 fixed guesses were no better than 3; turning off context checkpoints didn't help.
- The earlier "0.9 tok/s" came from the test program's median; `llama-cli` reports an average including warm-up
  tokens, which is why its baseline read 0.6.
- Safety: both hooks now also check the tensor's size, so a second model in the process (the helper) can't be
  mistaken for the big one's tensors of the same name.

`moe_run.py` now does all of this automatically: `--no-warmup`, packs the dense weights when they must stream, and
uses a same-family 0.8B/2B helper found next to the model (reserving its memory). End to end on the simulated 8 GB
machine, the 122B story prompt ran at **2.5 tok/s**.

## 12. Real 8 GB machine: a VM (2026-09-24, 16:35-16:48)

WSL2 (Ubuntu 26.04) VM capped at **8 GB RAM, 4 CPUs, no swap**, CPU-only build of our llama.cpp, models copied
into the VM's own disk, everything run through `moe_run.py` (the one-command tool). Log: `results/pc/vm-8gb-test.log`.

| Model | Prompt | Without helper | With helper (Qwen3.5-0.8B) | Lowest free RAM |
|---|---|---|---|---|
| 122B Q5 (91.5 GB) | explanation | 1.1 tok/s | **1.8 tok/s** | 1.6-1.9 GB |
| | code | 0.7 | **1.0** | 1.6-1.9 GB |
| | story | 1.1 | **1.7** | 1.6-2.0 GB |
| 35B Q5 (26 GB) | explanation / story | | **7.4 / 7.6 tok/s** | 4.4 GB |

- Zero out-of-memory kills. The tool's plan put 3.5-4.2 GB of always-needed weights in RAM (Linux leaves ~7.4 GB
  free on 8 GB, more than Windows' ~5.5).
- Estimates: 1.0-1.1 predicted vs 0.7-1.1 measured without the helper; 1.8-2.2 vs 1.0-1.8 with it (the helper
  factor of 2 is optimistic for code).
- Still flattering: the VM runs on the PC's fast cores and its ~5-6 GB/s NVMe (through the VM's virtual disk). A
  budget laptop with a ~3 GB/s drive and slower cores should be about half: roughly 0.5-1 tok/s for the 122B and
  ~4 tok/s for the 35B.

### 12b. Same VM with the disk capped at 3 GB/s (a typical PCIe 3 laptop SSD)

Cap applied inside the VM with cgroup v2 `io.max` (`rbps=3000000000`), which limits every read, including model
loading. Measured while busy: 2.3-3.0 GB/s average, 3.1-3.5 GB/s peaks. Log: `results/pc/vm-8gb-3gbs-test.log`.

| Model | Prompt | Without helper | With helper | Tool's estimate (without / with) |
|---|---|---|---|---|
| 122B Q5 | explanation | 0.7 tok/s | **1.1** | 0.7 / 1.2 |
| | code | 0.7 | **0.8** | 0.7 / 1.2 |
| | story | 0.7 | **1.1** | 0.7 / 1.2 |
| 35B Q5 | explanation / story | | **5.5 / 6.2** | 5.8-5.9 |

No out-of-memory kills (lowest free memory 1.6 GB for the 122B). Prompt reading: ~2 tokens/s for the 122B, 8-9 for
the 35B. The CPU is still the PC's (faster than a budget laptop's), but at these speeds the disk is the bottleneck:
it sat at ~2.9 GB/s, its cap, for the whole run.

Also tried: compressing expert data (zlib and lzma, 92 MB sample): 0.5% smaller. Quantized weights are effectively
incompressible, so reading compressed experts can't help.

## 13. Using fewer experts per token (opt-in `--experts K`, changes the output)

The router normally picks 8 experts per token. Forcing fewer (`--override-kv qwen35moe.expert_used_count=int:K`)
reads fewer bytes but changes the answers. Quality measured on the 122B with `llama-perplexity` (WikiText-2 test,
8 x 512 tokens), comparing each setting's token probabilities to the normal 8 experts (logs: `results/pc/kld/`):

| Experts per token | Expert reads saved | Same top token as 8 experts | Perplexity | Mean KL divergence |
|---|---|---|---|---|
| 8 (normal) | | | 3.518 | |
| 7 | 12.5% | 93.1% | 3.536 (+0.5%) | 0.033 |
| 6 | 25% | 88.3% | 3.645 (+3.6%) | 0.082 |
| 5 | 37.5% | 84.2% | 3.940 (+12%) | 0.169 |

For scale, a mean KLD of ~0.03 is roughly what dropping one quantization level (e.g. Q5 to Q4) costs; 0.08-0.17 is
a clearly weaker model. Speed it buys on the 8 GB / 3 GB/s VM (122B, story prompt, with the helper; log `results/pc/vm-8gb-3gbs-experts.log`):

| Experts per token | Generation | Prompt reading |
|---|---|---|
| 8 | 1.1 tok/s | 1.8 tok/s |
| 7 | 1.2 (+9%) | 2.2 |
| 6 | 1.4 (+27%) | 2.6 |

A real trade-off, not a free win: 7 costs little and gains little; 6 gains ~27% for a noticeably weaker model.
`moe_run.py --experts K` exposes it, off by default, and prints that answers will differ.

## 14. Guessing ahead on the budget laptop: MTP head vs helper, and a correction

Setup: the 8 GB / 4 CPU / no-swap VM with its disk capped at 3 GB/s (cgroup `io.max`). Qwen3.5 ships "MTP"
versions with a built-in multi-token-prediction layer (Unsloth UD-Q5_K_M builds; the 35B is 27 GB, 41 layers = 40 +
the MTP layer). llama.cpp runs it with `--spec-type draft-mtp`. Log: `results/pc/vm-mtp35.log`.

35B (its always-needed weights fit in RAM; 2 GB expert cache):

| Setup | Story | Code |
|---|---|---|
| No guessing | 5.1 tok/s | 5.3 |
| 0.8B helper (12 guesses, p >= 0.8) | 5.7 | 4.7 |
| MTP, 2 guesses | 5.1 | 5.0 |
| MTP, 3 guesses | 5.0 | 4.6 |
| MTP, 4 guesses | 4.6 | 4.4 |
| MTP, 3 guesses, p >= 0.5 | 5.4 | 5.2 |

Guessing doesn't help the 35B. Checking k guessed tokens costs the union of their experts, and when the
always-needed weights are already in RAM there's nothing to amortize. It paid off for the 122B only because one pass
reads the streamed always-needed weights once for several tokens. `moe_run` now only guesses when the always-needed
weights are streamed, and prefers a built-in MTP head over a helper (no extra RAM).

**Correction: guessing is not guaranteed to give identical output.** In the VM both MTP and the helper changed the
reply, identically, from the ~9th word. Tested with the same build and our cache on vs off: our cache gave
byte-identical output in both modes, and plain llama.cpp alone differed between guessing and not guessing. So it's
llama.cpp's batched verification rounding differently (a close call flips), not our code. On the Windows build every
greedy comparison happened to match. Honest statement: our cache never changes output; guessing gives the same model
at the same quality, but an occasional word can differ.

Also measured: plain llama.cpp with no cache ran the 35B in this VM at 0.5-0.6 tok/s vs 5.1 with our cache.

122B MTP (Unsloth UD-Q5_K_M, 93.6 GB, 49 layers = 48 + MTP), always-needed weights streamed (4.0 GB budget),
0.5 GB expert cache. Log: `results/pc/vm-mtp122.log`.

| Setup | Story | Code |
|---|---|---|
| No guessing | 0.6 tok/s | 0.6 |
| 0.8B helper | **1.2** | 0.8 |
| MTP, 2 guesses | 0.9 | 0.9 |
| MTP, 3 guesses, p >= 0.5 | 1.0 | 0.9 |
| MTP, 3 guesses, p >= 0.5, helper's RAM given to always-needed weights (4.6 GB) | 1.1 | |

MTP and the helper are about even: MTP is steadier across kinds of text (~+50% on both), the helper is better on
prose and worse on code. Either roughly doubles prose speed for the 122B on a budget laptop. Neither changes the
basic limit: ~4 GB read per token from a 3 GB/s drive.

## 15. Q8 on the 8 GB machine (2026-09-25)

Q8 is the highest common quantization (close to the original model). The same 8 GB / 4 CPU / no-swap VM with the disk
capped at 3 GB/s, the released Linux `moe`, 35B, 3 prompts x 128 tokens (`vm/q8-test.sh`):

| Prompt | 35B Q5 (26.2 GB) | 35B Q8 (36.9 GB) |
|---|---|---|
| Sky is blue | 8.5 tok/s | 5.7 tok/s |
| Python function | 7.6 | 5.1 |
| Mystery story | 8.6 | 6.7 |
| Average | 8.2 | 5.8 |

Q8 reads ~1.7x more expert data per token and runs at ~70% of Q5's speed: still faster than reading speed on a
cheap laptop. Lowest free memory during the runs was ~4 GB and there were no OOM kills. This VM runs nothing else
(7.3 GB free), so a real 8 GB Windows laptop with a browser open will be somewhat slower.

The 122B's always-needed weights are already Q8 inside the Q5_K_M file (5.84 GB either way, read from the GGUF
headers over HTTP), so going from Q5 to Q8 only grows the experts (84.9 -> 123.2 GB). On 8 GB, where those
always-needed weights are streamed anyway, the 122B should pay much less than 1.45x for Q8 (section 18).

## 16. Cache-aware routing (opt-in `--fast`, changes the output)

When the expert the model would pick isn't in RAM but a nearly-as-good one is, use the one in RAM. It is implemented
as a hook just before llama.cpp's top-k expert pick (`llama_set_moe_select_hook`): experts in the cache get their
router score multiplied by (1 + bonus), which only changes *which* experts are picked. The weights that mix the
chosen experts still come from the real scores. It applies only while generating (batches of 16 tokens or fewer),
not to prompt reading. `MOE_CACHE_BONUS=b`.

Quality and disk reads: 35B Q5 with a 2.7 GB expert cache (the 8 GB laptop plan), WikiText-2 4 x 512 tokens fed one
token at a time (`-ub 1`, so the cache state is what it would be while chatting), KL divergence against the same
setup without the bonus (logs: `results/route/` on the PC, `pc/route-sweep.ps1`):

| Bonus | Expert data read | Picks switched | Same top token | Mean KLD |
|---|---|---|---|---|
| none | 100% | 0% | 100% | 0 |
| 0.1 | 89% | 4.9% | 96.5% | 0.010 |
| 0.25 | 78% | 9.9% | 95.2% | 0.016 |
| 0.5 | 68% | 14.5% | 94.0% | 0.023 |
| **1.0** | **61%** | 18.0% | 93.1% | **0.029** |
| 2.0 | 57% | 20.1% | 92.2% | 0.038 |

For comparison, using 7 of 8 experts (section 13, measured on the 122B) saves 12.5% of reads at KLD 0.033.
Routing with bonus 1.0 saves 39% at 0.029, so it is the better way to trade a little quality for speed. The PC SSD
wait time dropped in step: 216 s -> 92 s for the 2,048 tokens. `--fast` uses bonus 1.0.

Keeping the model's own top-ranked pick regardless (`MOE_CACHE_PROTECT=1`) changed almost nothing (bonus 1.0:
KLD 0.028 vs 0.029, same reads), and keeping the top two (`MOE_CACHE_PROTECT=2`) didn't either (0.031). The top
picks are rarely the ones that get switched, so the plain bonus stays.

When it helps (35B Q5, one token at a time, 512 tokens, `pc/route-threshold.ps1`), by expert cache size measured in
tokens' worth of experts (0.74 GB per token): 0.5x (0.37 GB) no switches and no saving, because the previous token's
experts for a layer are already evicted when that layer routes; 1x (0.75 GB) 45% fewer reads (24% of picks switched);
2x (1.5 GB) 39% fewer; 3.6x (2.7 GB, the table above) 39% fewer. The 122B Q8 on 8 GB (0.13x) gained nothing (section 18).
So `--fast` turns this part on only when the cache holds at least one token's worth of experts.

Prior art note: colibri (github.com/JustVugg/colibri) deliberately never changes routing. Here it is opt-in and the
app says the answers will differ.

## 17. Slim mode: a model needs ~1x its size on disk, not 2x

Packing keeps a second copy of the experts laid out for fast reads. Slim mode frees the original file's copy of the
experts once they are packed, by punching holes in the file (`sparse.py`: `fallocate(PUNCH_HOLE)` on Linux,
`F_PUNCHHOLE` on macOS, `FSCTL_SET_SPARSE` + `FSCTL_SET_ZERO_DATA` on Windows). The file keeps its size and layout, so
llama.cpp still loads it, and the freed ranges read back as zeros. Packing works one layer at a time: build the
layer, write it and fsync, read back a sample and compare, then free that layer in the original and record it in
`<out>.progress`, so an interrupted setup resumes instead of losing data. Peak disk use is the model plus one layer.

The engine must then never read experts from the original file. The one path that did (an expert that doesn't fit a
small cache during a big prompt op, the "overflow" fallback) now points llama.cpp into a read-only mapping of the
packed file instead.

Test (VM, 35B Q5, `vm/slim-test.sh`): re-packed through the app with `--slim`. The model file went from 25 GB to 3 GB on
disk, and the packed copy is 22 GB, so the model uses ~25 GB instead of ~47. A fixed prompt with greedy sampling gave
byte-identical answers before and after slimming, with a 6 GB cache and with a 0.3 GB cache (12 experts overflowed
and were read through the packed mapping). The app slims models it downloaded itself automatically; a model file
the user brought is only slimmed with `--slim`.

## 18. The 122B at Q8 on the 8 GB machine (2026-09-25)

Qwen3.5-122B-A10B Q8_0 (129.9 GB, `unsloth/Qwen3.5-122B-A10B-GGUF`), packed with `--slim`: 123.2 GB of experts
packed and verified layer by layer in 275 s, and 123.2 GB freed in the original files (GGUFs 122 GB -> 7 GB on disk),
so the model uses 122 GB instead of 245. Then the 8 GB / 4 CPU / no-swap VM with the disk capped at 3 GB/s, 96 tokens
per run (`vm/q122-test.sh`; tok/s):

| Prompt | Plain | Helper 0.8B | `--fast` | Helper + `--fast` |
|---|---|---|---|---|
| Sky is blue | 0.6 | 0.6 | 0.6 | 0.6 |
| Python function | 0.5 | 0.8 | 0.6 | 0.8 |
| Mystery story | 0.6 | 0.5 | 0.6 | 0.5 |

- The 122B at Q8 runs on 8 GB at ~0.6 tok/s, versus ~0.7 at Q5 (section 12): about 15% slower for 1.45x bigger
  experts, because the streamed always-needed weights (the same 5.84 GB at Q5 and Q8) dominate. Lowest free memory
  1.6 GB, no OOM kills.
- `--fast` does nothing here. The 8 GB plan leaves the 122B a 0.5 GB expert cache (~50 experts of 10 MB), so there is
  rarely a cached alternative to switch to. It pays off when the cache holds a real share of the experts (the 35B:
  ~1,000).
- The helper model gains 0-45% at Q8 versus up to 2x at Q5. Verifying k guessed tokens needs the union of their experts,
  and at Q8 each extra expert costs 1.45x more to read, eating into what one shared pass over the dense weights saves.
- At 3 GB/s this is near the floor for this approach: ~5.4 GB read per token is ~1.8 s. What moves it: a faster SSD
  (~1.3 tok/s at 7 GB/s), 16 GB of RAM (always-needed weights stay in memory and the expert cache grows), or reading
  fewer bytes per token.

## 19. 4 GB machines, and tuning the 8 GB split (2026-09-25)

**Memory split for the 122B Q8 on 8 GB** (3 GB/s, `vm/split-test.sh`): expert cache + streamed always-needed weights of
0.3+4.9, 0.5+4.7 (the default), 1.0+4.2, 1.5+3.7, 2.2+3.0, 1.2+4.7 and 0.5+5.4 GB all gave 0.4-0.6 tok/s, and the default
is as good as any. The expert data read was identical (242.6 GB per run) for every cache size from 0.3 to 2.2 GB: one Q8
token of the 122B needs 3.85 GB of experts, more than any cache that fits next to the dense weights, so nothing is
reused. The drive ran at ~2.75 of its 3 GB/s. That is the floor for this machine.

**The 35B on a 4 GB machine** (4 GB / 4 CPU / no swap VM, ~3.2 GB free, 3 GB/s; `vm/ram4-probe.sh`, `vm/ram4-app.sh`).
The planner (sized for 8 GB: 1.0 GB margin + 1.2 GB of buffers) refused to run. Measured by hand with a 2k context and
batch 64:

| Setup | Peak process memory | Lowest free | tok/s |
|---|---|---|---|
| cache 0.3 + stream 0.8 | 1.36 GB | 2.0 GB | 1.3 |
| cache 0.3 + stream 1.4 | 1.93 GB | 1.4 GB | 1.8 |
| cache 0.5, dense left to the OS | (page cache) | | 3.1 |
| cache 1.0, dense left to the OS | | | 0.5 (thrash) |

Through the app, leaving the dense weights to the OS thrashed (0.4-0.5 tok/s) because the app's own process tips it
over, and a larger stream budget that left only ~0.45 GB free also collapsed (0.6-0.7). With less than ~1.3 GB truly
free, the OS evicts llama.cpp's small mapped tensors and refaults them. The planner's small-machine mode (under 5 GB
free): 2k context, batch 64, 0.5 GB of buffers, 0.2 GB minimum cache, the 1 GB margin kept, always stream unless there
is a clear surplus, and no helper model (its RAM is worth more to the streamed weights). Result through the app:
**1.9-2.0 tok/s on all three prompts**, lowest free 1.2 GB, no OOM. Below ~2.1 GB free it declines instead of thrashing.
A 4 GB Windows laptop (Windows itself uses ~2 GB) will usually have less than that free; a 4 GB Linux machine or
Chromebook is the realistic target.

## 20. Faster first word: batch-aware routing while reading the question (in `--fast`, changes the output)

Reading a question runs all of its tokens through each layer together, so each layer reads the union of the experts
the tokens picked: ~100 of 256 for a chat-sized question. On the 8 GB setup that's what makes the 122B take ~30 s to
read a 40-token question (the drive is saturated). Batch-aware routing applies the cache-aware idea to the batch: an
expert that two or more tokens already picked, or that is in RAM, gets a score bonus (1 + b), so near-ties collapse
onto experts that are read anyway. Only which experts are picked changes; the weights come from the real scores. It
applies to batches of more than 16 tokens (`MOE_BATCH_BONUS=b`).

Quality and reads: 122B Q5 on the PC, question-sized batches (48 tokens), WikiText-2 3 x 512, KLD against no bonus
(`pc/batch-sweep.ps1`, logs `results/batchroute/` on the PC). Every token here is read in batch mode, so this
overstates the effect on a chat, where only the question is:

| Bonus | Distinct experts per batch | Expert data read | Same top token | Mean KLD | Wall time |
|---|---|---|---|---|---|
| none | 106.6 | 100% | 100% | 0 | 347 s |
| 0.25 | 88.2 | 83% | 98.0% | 0.008 | 285 s |
| 0.5 | 80.1 | 75% | 97.5% | 0.009 | 266 s |
| **1.0** | **73.7** | **69%** | **96.2%** | **0.016** | **249 s** |
| 2.0 | 69.6 | 65% | 95.4% | 0.021 | 242 s |

Time to first word, 122B Q8 on the 8 GB / 3 GB/s VM, two chat questions, the released app (`vm/ttfw-test.sh`):

| Question | Question reading, normal | with bonus 1.0 | Whole run (load + question + 8 tokens) |
|---|---|---|---|
| Japan trip (~40 tokens) | 1.3 tok/s | 2.3 tok/s | 67 s -> 53 s |
| Roth vs traditional IRA | 1.2 tok/s | 2.1 tok/s | 62 s -> 51 s |

Question reading got ~75% faster for 33% fewer distinct experts. The drop from ~100 to ~67 experts per layer lets the
batch fit in the 8 GB plan's 0.5 GB cache (~72 slots), so the overflow path (reads through the packed mapping) is no
longer needed. `--fast` now turns this on together with cache-aware routing (section 16); the default stays exact.

## 21. The 122B on a 16 GB machine (2026-09-25)

122B Q8, 16 GB / 4 CPU / no-swap VM, disk capped at 3 GB/s, the released app (v0.2.3), 96 tokens (`vm/ram16-test.sh`).
The plan keeps the always-needed weights in RAM and gives the experts a 7.3 GB cache:

| Run | Question reading | Answering |
|---|---|---|
| plain (sky, story) | 1.7-1.8 tok/s | 1.4 tok/s |
| `--fast` (sky, story) | 1.9-2.0 | **1.9-2.0** |
| helper model | 1.8 | 1.3 (the app doesn't guess when dense fits in RAM) |

Twice the 8 GB speed (0.6), and `--fast` adds ~40%: its answering part switched 18-21% of picks to experts already in
the 7.3 GB cache, which is ~1.9 tokens' worth of Q8 experts, above the 1x threshold (section 16). No OOM kills.

## 22. OpenAI gpt-oss-20b and gpt-oss-120b (2026-09-25)

Both are MoE models in their native MXFP4 format (`ggml-org/gpt-oss-*-GGUF`), and both are light per token: the 120b
has 60.9 GB of experts but only 1.7 GB of always-needed weights (the Qwen 122B has 5.8), 4 of 128 experts per token.
Support needed three fixes: their experts carry small per-expert bias tensors (`*_exps.bias`), which must not be
packed or freed (only `*_exps.weight` is), their router ranks experts by raw logits (softmax after top-k), so a bonus
is added as ln(1 + b) instead of multiplying, and they always reason first, so the app sets `reasoning_effort` to
low unless `--think` is given.

Exactness: the same greedy answer from plain llama.cpp and from stowaway, packed and slimmed, with a 4 GB cache and a
0.3 GB one (252 experts overflowed): identical. Slimming the 20b: 11.5 GB -> 1.9 GB plus 9.7 GB packed.

Speed (VMs capped at 3 GB/s, the app, 128 tokens, 3 prompts; `vm/gptoss-speed.sh`):

| | 4 GB | 8 GB | 16 GB |
|---|---|---|---|
| gpt-oss-20b | 2.3-2.4 tok/s | 6.7-9.3 | |
| gpt-oss-120b | 0.7 (small-machine mode, streaming; lowest free 1.2 GB) | 2.1-2.5 | |
| gpt-oss-120b with cache-aware routing (bonus 1.0) | | 4.1-4.6 | 4.1-4.9 |

**But gpt-oss is far more sensitive to expert swaps.** gpt-oss-20b, one token at a time, 2 x 512 WikiText, the 8 GB
plan's 3.5 GB cache (`vm/gptoss-kld*.sh`):

| Cache-aware bonus | Picks switched | Same top token | Mean KLD |
|---|---|---|---|
| none, hook on (control) | 0% | 100% | 0.000001 |
| 0.25 | 4.9% | 82.2% | 0.090 |
| 1.0 | 8.2% | 73.5% | 0.141 |
| 1.0, only the 4th pick swappable (`MOE_CACHE_PROTECT=3`) | 4.5% | 81.8% | 0.074 |

gpt-oss uses only 4 experts per token, so each one carries more of the output. (Its baseline perplexity on raw
WikiText is ~230, because it's trained on its chat format, which also inflates KL divergence; the gap to Qwen is still
large.) So `--fast` is not used for gpt-oss models: the app says so and runs them exactly. gpt-oss-120b at ~2.3 tok/s is
still the best big model for 8 GB, about 4x the Qwen 122B at Q8 (section 18).

**A measurement fix found on the way:** with the hook on but nothing swapped, output still drifted a little (KLD 0.004
on Qwen, 0.015 on gpt-oss). The hook could change the *order* of the chosen experts, which changes the order their
outputs are summed in, and so the rounding. The hook now writes scores so that the chosen set keeps the model's own
order, which makes a zero-swap run bit-identical. Re-measured Qwen 35B with the fix: bonus 1.0 KLD 0.028 / 93.3% same top
token (was 0.029 / 93.1%), bonus 0.5 0.022 / 94.9%. The conclusions of sections 16 and 20 stand.

## What didn't work, and why
- Sharing experts inside the helper's guess-checking batches (`MOE_BATCH_VERIFY=1`, 2-16 token batches; 122B Q8, 8 GB,
  3 GB/s, 3 prompts): answering 0.6/0.9/0.5 tok/s vs 0.6/0.8/0.5 without it. The batches only touch 15-26 experts
  per layer, so there's little to share, and each pass is dominated by the streamed always-needed weights. It stays an
  opt-in switch and isn't part of `--fast`. The question-reading part was again +60-80% in the same runs
  (1.1-1.3 -> 1.9-2.3 tok/s).

- **Windows PrefetchVirtualMemory called inline** made things 3× slower (it blocks while walking the range). Moving
  it to a helper thread fixed the slowdown, but it's still only a hint.
- **Pre-gating v1** (hidden state grabbed through llama.cpp's eval callback) lost: the GPU pauses cost more than the
  early loading saved. v2 fixed that (section 5).
- **Smarter eviction policies** (SLRU, 2Q, ARC, LFU-decay): no better than LRU on this workload.
- **Locking the cache in RAM** and **fewer/more I/O threads** on the unpacked layout: no real change; the
  3-pieces-per-expert layout was the limit.
- **Co-occurrence prediction** (which experts tend to follow which) was too weak to be useful (~45%).

## Next steps, in order

1. More RAM: 64 GB would let the cache hold ~half the experts (hit rate ~70% -> ~85-88% in the simulations), cutting
   reads per token by more than half.
2. Turn on XMP/EXPO in the BIOS: the RAM is a DDR5-6000 kit running at 4800 (user has to do this).
3. Pre-gating with a confidence threshold instead of a fixed count.
4. Try the cache with more experts on the GPU (the always-needed 6.6 GB leaves ~7 GB of VRAM).
5. Topic-aware warm start: code prompts use the most distinct experts (JS divergence 0.25-0.33).
6. Clean Mac measurement with `sudo purge` between runs.
