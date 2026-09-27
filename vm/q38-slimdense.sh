#!/bin/bash
# Does shrinking Qwen3.8's 8-bit always-needed weights to Q5_1 cost quality? Reference = the unchanged model (batch
# mode, 4 x 512 WikiText tokens, full cache). Then file 3 of 3 (layers 14-47, ~70% of those weights, no n-gram
# table) is rewritten with slim_dense.py and compared. The original file 3 is kept as .q8orig.
cd /root/moe && SRC=/mnt/c/Users/FSociety/moe-router-study && cp $SRC/{slim_dense.py,sparse.py,moe.py} .
cp $SRC/dist/linux-test/llama-perplexity app/ && chmod +x app/llama-perplexity
D=models/q38-iq4xs; M=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf; F3=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf
R=results/q38-slimdense; mkdir -p $R; W=$SRC/data/wikitext/wikitext-2-raw/wiki.test.raw
ARGS="-m $M -ngl 0 --no-repack --no-op-offload -t 4 -c 512 -b 512 -ub 512 --chunks 4 -f $W --no-warmup"
export MOE_CACHE_GB=8 MOE_IO_THREADS=4 MOE_STATS=1 EXPERT_CACHE_PACKED=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-experts-packed EXPERT_CACHE_CHUNK_KB=8192 LLAMA_NO_MMAP_PREFETCH=1
show() { grep -hE "Final estimate|Mean    KLD|Same top p:|99.0%   KLD|failed|error" $1 | sed 's/^[0-9.]* I //;s/^/  /' | cut -c1-150; }
if [ ! -f $R/base.kld ]; then
  echo "=== reference (unchanged model)  $(date +%T)"
  app/llama-perplexity $ARGS --kl-divergence-base $R/base.kld > $R/base.txt 2>&1; echo "  exit $?"; show $R/base.txt
fi
if [ ! -f $F3.q8orig ]; then
  echo "=== rewrite file 3 with Q5_1  $(date +%T)"
  python3 slim_dense.py $F3 --out $F3.q5 --type q5_1 && mv $F3 $F3.q8orig && mv $F3.q5 $F3
  rm -f $D/*dense-packed*  # always-needed weights changed: the streaming copy is rebuilt on the next run
  du -sh $F3 $F3.q8orig
fi
echo "=== Q5_1 file 3 vs reference  $(date +%T)"
app/llama-perplexity $ARGS --kl-divergence-base $R/base.kld --kl-divergence > $R/q51.txt 2>&1; echo "  exit $?"; show $R/q51.txt
# how far can it go? Q4_1 (5 bits) the same way, then back to Q5_1 for the speed runs
python3 slim_dense.py $F3.q8orig --out $F3.q41 --type q4_1 && mv $F3 $F3.q51 && mv $F3.q41 $F3
echo "=== Q4_1 file 3 vs reference  $(date +%T)"
app/llama-perplexity $ARGS --kl-divergence-base $R/base.kld --kl-divergence > $R/q41.txt 2>&1; echo "  exit $?"; show $R/q41.txt
rm -f $F3 && mv $F3.q51 $F3
