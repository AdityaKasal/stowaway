#!/bin/bash
# The real test: moe_run on an 8 GB / 4-CPU Linux VM with no GPU and no swap.
cd /root/moe
echo "=== machine"; free -m | head -2; nproc; grep -m1 "model name" /proc/cpuinfo
low_mem() {  # lowest MemAvailable while a command runs
  local lo=999999; while kill -0 $1 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done; echo "lowest free memory during the run: ${lo} MB"; }
run() {  # name model packed draft prompt
  local name=$1 model=$2 packed=$3 draft=$4 prompt=$5
  echo; echo "=== $name  ($(date +%T))"
  python3 moe_run.py $model --packed $packed --draft $draft --threads 4 -n 96 -p "$prompt" > /tmp/$name.out 2> /tmp/$name.err &
  local pid=$!; low_mem $pid & local mp=$!; wait $pid; echo "exit $?"; wait $mp
  grep -E "^(model|machine|plan|helper):" /tmp/$name.out
  grep -oE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/$name.out
  grep -E "dense streaming: [0-9.]+ GB pinned|error|failed|Killed" /tmp/$name.err | head -3
}
P1="Explain in a short paragraph why the sky is blue."
P2="Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring."
P3="Write the opening two sentences of a mystery story set in a lighthouse."
M122=models/122b/Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf
for i in 1 2 3; do eval p=\$P$i
  run "122b-p$i-helper"   $M122 models/122b/experts-packed models/Qwen3.5-0.8B-Q4_K_M.gguf "$p"
  run "122b-p$i-nohelper" $M122 models/122b/experts-packed none "$p"
done
for i in 1 3; do eval p=\$P$i
  run "35b-p$i-helper"    models/Qwen3.5-35B-A3B-Q5_K_M.gguf models/35b-packed models/Qwen3.5-0.8B-Q4_K_M.gguf "$p"
done
echo; echo "=== out-of-memory kills during the test:"; dmesg 2>/dev/null | grep -ciE "out of memory|oom-kill" || true
