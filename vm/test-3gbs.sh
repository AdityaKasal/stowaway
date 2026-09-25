#!/bin/bash
# Same test as test.sh, but the VM's disk is capped at 3 GB/s for reads (a typical PCIe 3 laptop SSD).
cd /root/moe
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
MAJMIN=$(cat /sys/block/$DEV/dev)
CAP=3000000000
if [ -f /sys/fs/cgroup/cgroup.controllers ] && grep -qw io /sys/fs/cgroup/cgroup.controllers; then
  echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null
  mkdir -p /sys/fs/cgroup/moe3g
  echo "$MAJMIN rbps=$CAP" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs \
    && echo "disk cap: cgroup io.max on $DEV ($MAJMIN) = $(cat /sys/fs/cgroup/moe3g/io.max)"
fi
if ! grep -q rbps /sys/fs/cgroup/moe3g/io.max 2>/dev/null; then
  export EXPERT_CACHE_MAX_MBPS=3000; echo "disk cap: cgroup io.max not available, using the cache's own 3000 MB/s limiter"
fi
# peak and average read speed of the disk while a command runs (to prove the cap)
disk_rate() {
  local pk=0 tot=0 n=0 prev=$(awk -v d=$DEV '$3==d {print $6}' /proc/diskstats)
  while kill -0 $1 2>/dev/null; do sleep 1; cur=$(awk -v d=$DEV '$3==d {print $6}' /proc/diskstats)
    r=$(( (cur - prev) * 512 / 1000000 )); prev=$cur; [ $r -gt $pk ] && pk=$r; [ $r -gt 50 ] && { tot=$((tot + r)); n=$((n + 1)); }; done
  echo "disk reads: peak ${pk} MB/s, average $(( n ? tot / n : 0 )) MB/s while busy"; }
low_mem() { local lo=999999; while kill -0 $1 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done; echo "lowest free memory: ${lo} MB"; }
run() {
  local name=$1 model=$2 packed=$3 draft=$4 prompt=$5
  echo; echo "=== $name  ($(date +%T))"
  python3 moe_run.py $model --packed $packed --draft $draft --threads 4 -n 96 -p "$prompt" > /tmp/$name.out 2> /tmp/$name.err &
  local pid=$!; low_mem $pid & local m1=$!; disk_rate $pid & local m2=$!; wait $pid; echo "exit $?"; wait $m1 $m2
  grep -E "^(plan|helper):" /tmp/$name.out
  grep -oE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/$name.out
  grep -E "error|failed|Killed" /tmp/$name.err | head -3
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
  run "35b-p$i-helper" models/Qwen3.5-35B-A3B-Q5_K_M.gguf models/35b-packed models/Qwen3.5-0.8B-Q4_K_M.gguf "$p"
done
echo; echo "=== out-of-memory kills:"; dmesg 2>/dev/null | grep -ciE "out of memory|oom-kill" || true
