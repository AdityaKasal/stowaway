#!/bin/bash
# Qwen3.5-122B: how much quality does each size lose? Reference = Q8 (in the VM). Download UD-IQ4_XS straight into the
# packed layout, then KLD of IQ4_XS vs Q8 on WikiText (batch mode, 4 x 512 tokens).
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/{moe.py,streampack.py,sparse.py,repack_experts.py,pack_dense.py} .
cp /mnt/c/Users/FSociety/moe-router-study/dist/linux-test/llama-perplexity app/ && chmod +x app/llama-perplexity
W=/mnt/c/Users/FSociety/moe-router-study/data/wikitext/wikitext-2-raw/wiki.test.raw; R=results/q122-quality; mkdir -p $R
export STOWAWAY_NO_UPDATE_CHECK=1 MOE_IO_THREADS=6 MOE_STATS=1 LLAMA_NO_MMAP_PREFETCH=1 EXPERT_CACHE_CHUNK_KB=8192
# 1. the reference logits from Q8 (while the download runs)
Q8=models/122b-q8/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf
if [ ! -f $R/base-q8.kld ]; then
  echo "Q8 reference $(date +%T)"
  MOE_CACHE_GB=8 EXPERT_CACHE_PACKED=models/122b-q8/Qwen3.5-122B-A10B-Q8_0-experts-packed \
    app/llama-perplexity -m $Q8 -ngl 0 --no-repack --no-op-offload -t 8 -c 512 -b 512 -ub 256 --chunks 4 -f $W --no-warmup \
    --kl-divergence-base $R/base-q8.kld > $R/q8.txt 2>&1 &
  REF=$!
fi
# 2. download UD-IQ4_XS straight into the packed layout
python3 - <<'PY'
import moe, streampack
repo = "unsloth/Qwen3.5-122B-A10B-GGUF"
files = [f"UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
exp = [moe.hf_checksum(repo, f) for f in files]
d = "/root/moe/models/122b-iq4xs"
streampack.fetch_packed(repo, files, d, d + "/Qwen3.5-122B-A10B-UD-IQ4_XS-experts-packed", exp, "stowaway-research")
PY
echo "download done $(date +%T)"; du -sh models/122b-iq4xs
[ -n "$REF" ] && wait $REF; echo "reference done $(date +%T)"; grep -E "Final estimate" $R/q8.txt
# 3. IQ4_XS vs Q8
IQ=models/122b-iq4xs/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf
MOE_CACHE_GB=8 EXPERT_CACHE_PACKED=models/122b-iq4xs/Qwen3.5-122B-A10B-UD-IQ4_XS-experts-packed \
  app/llama-perplexity -m $IQ -ngl 0 --no-repack --no-op-offload -t 8 -c 512 -b 512 -ub 256 --chunks 4 -f $W --no-warmup \
  --kl-divergence-base $R/base-q8.kld --kl-divergence > $R/iq4.txt 2>&1
echo "IQ4_XS vs Q8:"; grep -hE "Final estimate|Mean    KLD|Same top p:|99.0%   KLD" $R/iq4.txt | sed 's/^[0-9.]* I //'
cp $R/base-q8.kld /mnt/c/Users/FSociety/moe-router-study/results/base-q8-122b.kld && echo "copied the reference for the Q5 test on Windows"
