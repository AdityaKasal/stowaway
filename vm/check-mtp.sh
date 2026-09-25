#!/bin/bash
cd /root/moe; mkdir -p /root/mtpcheck
export LLAMA_NO_MMAP_PREFETCH=1 EXPERT_CACHE_PACKED=models/35b-mtp-experts-packed EXPERT_CACHE_CHUNK_KB=8192 MOE_CACHE_GB=2 MOE_IO_THREADS=4 MOE_PREGATE=0
P="Write the opening two sentences of a mystery story set in a lighthouse."
run() { local name=$1; shift
  llama.cpp/build/bin/llama-cli -m models/Qwen3.5-35B-A3B-MTP-UD-Q5_K_M.gguf -ngl 0 --no-repack --no-op-offload -c 4096 -b 128 -ub 128 \
    -t 4 -tb 4 -n 96 --temp 0 --seed 1 -p "$P" -st --simple-io --no-display-prompt --no-warmup "$@" > /root/mtpcheck/$name.out 2>/dev/null; }
run none; run mtp2 --spec-type draft-mtp --spec-draft-n-max 2
run helper -md models/Qwen3.5-0.8B-Q4_K_M.gguf -ngld 0 --spec-type draft-simple --spec-draft-n-max 12 --spec-draft-p-min 0.8
# the reply is what comes after the prompt line ("> ...") and before the timing line
reply() { awk '/^> /{on=1; next} /\[ Prompt/{on=0} on' /root/mtpcheck/$1.out; }
for v in mtp2 helper; do
  if [ "$(reply none)" = "$(reply $v)" ]; then echo "$v: same reply ($(reply none | wc -w) words)"; else echo "$v: DIFFERENT reply"; diff <(reply none) <(reply $v) | head -8; fi
done
echo "--- start of the reply:"; reply none | head -6
