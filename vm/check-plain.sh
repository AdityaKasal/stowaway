#!/bin/bash
# Is the difference from our code or from llama.cpp itself? Same build, our cache on vs off, with and without MTP.
cd /root/moe; mkdir -p /root/mtpcheck
P="Write the opening two sentences of a mystery story set in a lighthouse."
run() { local name=$1; shift
  env "$@" true
  llama.cpp/build/bin/llama-cli -m models/Qwen3.5-35B-A3B-MTP-UD-Q5_K_M.gguf -ngl 0 --no-repack --no-op-offload -c 4096 -b 128 -ub 128 \
    -t 4 -tb 4 -n 48 --temp 0 --seed 1 -p "$P" -st --simple-io --no-display-prompt --no-warmup $SPEC > /root/mtpcheck/$name.out 2>/dev/null; }
SPEC="" run plain-none
SPEC="--spec-type draft-mtp --spec-draft-n-max 2" run plain-mtp2
export LLAMA_NO_MMAP_PREFETCH=1 EXPERT_CACHE_PACKED=models/35b-mtp-experts-packed EXPERT_CACHE_CHUNK_KB=8192 MOE_CACHE_GB=2 MOE_IO_THREADS=4 MOE_PREGATE=0
SPEC="" run moe-none
SPEC="--spec-type draft-mtp --spec-draft-n-max 2" run moe-mtp2
reply() { awk '/^> /{on=1; next} /\[ Prompt/{on=0} on' /root/mtpcheck/$1.out; }
cmp() { [ "$(reply $1)" = "$(reply $2)" ] && echo "$1 vs $2: same" || echo "$1 vs $2: DIFFERENT"; }
cmp plain-none moe-none; cmp plain-mtp2 moe-mtp2; cmp plain-none plain-mtp2; cmp moe-none moe-mtp2
grep -oE "Generation: [0-9.]+ t/s" /root/mtpcheck/*.out
