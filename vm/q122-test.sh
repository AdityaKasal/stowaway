#!/bin/bash
# 122B Q8 on the 8 GB / 4 CPU / no-swap VM, disk capped at 3 GB/s: plain, helper model, --fast routing, both
cd /root/moe
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
echo "disk cap: $(cat /sys/fs/cgroup/moe3g/io.max)"; free -g | head -2; nproc
M=models/122b-q8/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf
low_mem() { local lo=999999; while kill -0 $1 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done; echo "lowest free memory: ${lo} MB"; }
run() {
  local name=$1 prompt=$2; shift 2
  echo; echo "=== $name  ($(date +%T))"
  env "$@" app/moe run $M --threads 4 -n 96 -p "$prompt" $EXTRA < /dev/null > /tmp/$name.out 2> /tmp/$name.err &
  local pid=$!; low_mem $pid & local m=$!; wait $pid; local rc=$?; echo "exit $rc"; wait $m
  if [ $rc -ne 0 ]; then tail -5 /tmp/$name.err; echo "stopping: a run failed"; exit 1; fi
  grep -hE "^(plan|guessing|note):" /tmp/$name.out /tmp/$name.err
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/$name.out /tmp/$name.err
  grep -hiE "error|failed|Killed" /tmp/$name.err | head -3
}
P1="Explain in a short paragraph why the sky is blue."
P2="Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring."
P3="Write the opening two sentences of a mystery story set in a lighthouse."
for i in 1 2 3; do eval p=\$P$i
  EXTRA="--draft none" run "122q8-p$i-plain" "$p" X=1
  EXTRA=""             run "122q8-p$i-helper" "$p" X=1
  EXTRA="--draft none" run "122q8-p$i-fast" "$p" MOE_CACHE_BONUS=1.0
  EXTRA=""             run "122q8-p$i-helper-fast" "$p" MOE_CACHE_BONUS=1.0
done
echo; echo "=== out-of-memory kills: $(dmesg 2>/dev/null | grep -ciE 'out of memory|oom-kill')"
