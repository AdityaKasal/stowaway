#!/bin/bash
# Why did Q5_1 cost so much (RESULTS.md 29)? (a) the same rounding with the hyper-connection, shared-expert and ple_key
# weights left at 8 bits (as Unsloth does even at 2 bits); (b) Unsloth's own UD-Q2_K_XL always-needed weights (Q5_K/
# Q6_K with an importance matrix) spliced in. File 3 only, KLD vs the same reference; the original file 3 is restored.
cd /root/moe && SRC=/mnt/c/Users/FSociety/moe-router-study
cp $SRC/{slim_dense.py,splice_dense.py,sparse.py,streampack.py,repack_experts.py} .
D=models/q38-iq4xs; M=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf; F3=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf
R=results/q38-slimdense; W=$SRC/data/wikitext/wikitext-2-raw/wiki.test.raw
ARGS="-m $M -ngl 0 --no-repack --no-op-offload -t 4 -c 512 -b 512 -ub 512 --chunks 4 -f $W --no-warmup"
export MOE_CACHE_GB=8 MOE_IO_THREADS=4 MOE_STATS=1 EXPERT_CACHE_PACKED=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-experts-packed EXPERT_CACHE_CHUNK_KB=8192 LLAMA_NO_MMAP_PREFETCH=1
show() { grep -hE "Mean    KLD|Same top p:|99.0%   KLD|failed|error" $1 | grep -v common_fit | sed 's/^[0-9.]* I //;s/^/  /' | cut -c1-150; }
[ -f $F3.q8orig ] || { echo "no original file 3 kept: stopping"; exit 1; }
kld() { rm -f $F3 && mv $2 $F3 && echo "=== $1  $(date +%T)" && app/llama-perplexity $ARGS --kl-divergence-base $R/base.kld --kl-divergence > $R/$3.txt 2>&1; echo "  exit $?"; show $R/$3.txt; }
python3 slim_dense.py $F3.q8orig --out $F3.a --type q5_1 --keep hc_,shexp,ple_key && kld "(a) Q5_1, hc_/shexp/ple_key kept at 8 bits" $F3.a keep8
python3 splice_dense.py $F3.q8orig $F3.b unsloth/Qwen3.8-Flash-Next-GGUF UD-Q2_K_XL && kld "(b) UD-Q2_K_XL's always-needed weights" $F3.b q2kxl
rm -f $F3 && mv $F3.q8orig $F3 && rm -f $D/*dense-packed* && echo "restored the original file 3"; ls -la $D | grep 00003
