#!/bin/bash
# helper-model guessing on the 8 GB VM (122B Q8, 3 GB/s): does sharing experts inside guess-checking batches help?
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux/stowaway /mnt/c/Users/FSociety/moe-router-study/dist/linux/llama-* app/ && chmod +x app/*
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
M=models/122b-q8/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf
run() {
  echo "=== $1  $(date +%T)"; local p=$2; shift 2
  env STOWAWAY_NO_UPDATE_CHECK=1 MOE_STATS=1 "$@" app/stowaway run $M --threads 4 -n 96 -p "$p" < /dev/null > /tmp/v.out 2> /tmp/v.err
  echo "  exit $?"
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/v.out /tmp/v.err | sed 's/^/  /'
  grep -hE "batch-aware|generation only: read" /tmp/v.err | sed 's/^/  /' | cut -c1-150
}
P1="Explain in a short paragraph why the sky is blue."
P2="Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring."
P3="Write the opening two sentences of a mystery story set in a lighthouse."
for i in 1 2 3; do eval p=\$P$i
  run "p$i helper" "$p" X=1
  run "p$i helper+prompt-sharing" "$p" MOE_BATCH_BONUS=1.0
  run "p$i helper+prompt+verify-sharing" "$p" MOE_BATCH_BONUS=1.0 MOE_BATCH_VERIFY=1
done
