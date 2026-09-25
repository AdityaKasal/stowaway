#!/bin/bash
# quality cost of --fast's answering part on gpt-oss-20b: one token at a time, the 8 GB plan's 3.5 GB cache, KLD vs no bonus
cd /root/moe
M=models/gpt-oss/gpt-oss-20b-MXFP4.gguf; PK=models/gpt-oss/gpt-oss-20b-MXFP4-experts-packed; mkdir -p results/gptoss-kld
W=/mnt/c/Users/FSociety/moe-router-study/data/wikitext/wikitext-2-raw/wiki.test.raw
ARGS="-m $M -ngl 0 --no-repack --no-op-offload -t 8 -c 512 -b 512 -ub 1 --chunks 2 -f $W --no-warmup"
export MOE_CACHE_GB=3.5 MOE_IO_THREADS=4 MOE_PREGATE=6 MOE_STATS=1 EXPERT_CACHE_PACKED=$PK EXPERT_CACHE_CHUNK_KB=8192 LLAMA_NO_MMAP_PREFETCH=1
run() { echo "=== bonus $1  $(date +%T)"; MOE_CACHE_BONUS=$1 app/llama-perplexity $ARGS $2 > results/gptoss-kld/b$1.txt 2>&1; echo "  exit $?"
  grep -hE "Final estimate|Mean    KLD|Same top p:|generation only: read|cache-aware routing \(" results/gptoss-kld/b$1.txt | sed 's/^[0-9.]* I //;s/^/  /' | cut -c1-150; }
prun() { echo "=== bonus $1 protect $2  $(date +%T)"; MOE_CACHE_PROTECT=$2 MOE_CACHE_BONUS=$1 app/llama-perplexity $ARGS --kl-divergence-base results/gptoss-kld/base.kld --kl-divergence > results/gptoss-kld/b$1-p$2.txt 2>&1
  grep -hE "Mean    KLD|Same top p:|generation only: read|cache-aware routing \(" results/gptoss-kld/b$1-p$2.txt | sed 's/^[0-9.]* I //;s/^/  /' | cut -c1-150; }
prun 1.0 4; prun 0.25 0; prun 1.0 0; prun 1.0 3
