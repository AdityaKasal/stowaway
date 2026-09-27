#!/bin/bash
# Qwen3.8-Flash-Next UD-IQ4_XS: quality cost of --fast's answering part (cache-aware routing), one token at a time with
# the 16 GB plan's cache, KLD vs normal routing on WikiText (2 x 512 tokens)
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux-test/llama-perplexity app/ && chmod +x app/llama-perplexity
D=models/q38-iq4xs; M=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf; R=results/q38-kld; mkdir -p $R
W=/mnt/c/Users/FSociety/moe-router-study/data/wikitext/wikitext-2-raw/wiki.test.raw
ARGS="-m $M -ngl 0 --no-repack --no-op-offload -t 4 -c 512 -b 512 -ub 1 --chunks 2 -f $W --no-warmup"
export MOE_CACHE_GB=8.5 MOE_IO_THREADS=4 MOE_PREGATE=6 MOE_STATS=1 EXPERT_CACHE_PACKED=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-experts-packed EXPERT_CACHE_CHUNK_KB=8192 LLAMA_NO_MMAP_PREFETCH=1
show() { grep -hE "Final estimate|Mean    KLD|Same top p:|generation only: read|cache-aware routing \(|hit" $1 | sed 's/^[0-9.]* I //;s/^/  /' | cut -c1-150; }
echo "=== normal routing (reference)  $(date +%T)"
app/llama-perplexity $ARGS --kl-divergence-base $R/base.kld > $R/base.txt 2>&1; echo "  exit $?"; show $R/base.txt
for b in 1.0 0.25; do
  echo "=== cache bonus $b  $(date +%T)"
  MOE_CACHE_BONUS=$b app/llama-perplexity $ARGS --kl-divergence-base $R/base.kld --kl-divergence > $R/b$b.txt 2>&1; echo "  exit $?"; show $R/b$b.txt
done
