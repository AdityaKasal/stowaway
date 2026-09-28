#!/bin/bash
# How many predicted experts to preload for Qwen3.8 (MOE_PREGATE; the app uses 6), and 1 vs 2 layers ahead
cd /root/moe && export STOWAWAY_NO_UPDATE_CHECK=1 && cp /mnt/c/Users/FSociety/moe-router-study/moe.py . && mkdir -p app-prof && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux-prof/llama-cli app-prof/ && chmod +x app-prof/llama-cli
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
M=models/q38-iq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
L="Write the opening two sentences of a mystery story set in a lighthouse."
run() {
  echo "=== $1  $(date +%T)"; shift
  env "$@" STOWAWAY_LLAMA_ARGS="--temp 0" GGML_CPU_PROFILE=1 MOE_STATS=1 python3 moe.py run $M --bin app-prof --threads 4 -n 96 --draft none -p "$L" < /dev/null > /tmp/g.out 2> /tmp/g.err
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/g.out /tmp/g.err | sed 's/^/  /'
  grep -hE "expert hook on thread 0|generation only: read|lookups|pre-gating|guesses" /tmp/g.err | sed 's/^/  /' | cut -c1-170
}
run "pre-gating off" MOE_PREGATE=0
run "preload 10" MOE_PREGATE=10
