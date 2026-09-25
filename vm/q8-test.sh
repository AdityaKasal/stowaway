#!/bin/bash
# 35B Q5 vs Q8 on the 8 GB / 4 CPU / no-swap VM with the disk capped at 3 GB/s, using the released Linux moe.
cd /root/moe
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
MAJMIN=$(cat /sys/block/$DEV/dev)
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$MAJMIN rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
echo "disk cap: $(cat /sys/fs/cgroup/moe3g/io.max)"; free -g | head -2; nproc
low_mem() { local lo=999999; while kill -0 $1 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done; echo "lowest free memory: ${lo} MB"; }
run() {
  local name=$1 model=$2 prompt=$3
  echo; echo "=== $name  ($(date +%T))"
  app/moe run $model --threads 4 -n 128 -p "$prompt" < /dev/null > /tmp/$name.out 2> /tmp/$name.err &
  local pid=$!; low_mem $pid & local m=$!; wait $pid; echo "exit $?"; wait $m
  grep -E "^(model|machine|plan|guessing|note):" /tmp/$name.out /tmp/$name.err | sed 's/^[^:]*://' 
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/$name.out /tmp/$name.err
  grep -hiE "error|failed|Killed" /tmp/$name.err | head -3
}
P1="Explain in a short paragraph why the sky is blue."
P2="Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring."
P3="Write the opening two sentences of a mystery story set in a lighthouse."
for i in 1 2 3; do eval p=\$P$i
  run "35b-q5-p$i" models/Qwen3.5-35B-A3B-Q5_K_M.gguf "$p"
  run "35b-q8-p$i" models/Qwen3.5-35B-A3B-Q8_0.gguf "$p"
done
echo; echo "=== out-of-memory kills: $(dmesg 2>/dev/null | grep -ciE 'out of memory|oom-kill')"
