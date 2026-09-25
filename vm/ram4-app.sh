#!/bin/bash
# the updated planner (moe.py from source, released llama binaries) on the 4 GB VM at 3 GB/s
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/{moe.py,repack_experts.py,sparse.py,pack_dense.py} .
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
sync; echo 3 > /proc/sys/vm/drop_caches; free -m | head -2
run() {
  echo "=== $1  $(date +%T)"; shift
  python3 moe.py run "$@" --bin app --threads 4 -n 96 < /dev/null > /tmp/a4.out 2> /tmp/a4.err &
  local pid=$! lo=999999; while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done
  wait $pid; echo "  exit $?, lowest free ${lo} MB"
  grep -hE "^(machine|small|plan|guessing|note|fast):" /tmp/a4.out /tmp/a4.err | sed 's/^/  /' | cut -c1-160
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/a4.out /tmp/a4.err | sed 's/^/  /'
  grep -hiE "can't run|Killed|error" /tmp/a4.out /tmp/a4.err | head -3
}
Q5=models/Qwen3.5-35B-A3B-Q5_K_M.gguf; Q8=models/Qwen3.5-35B-A3B-Q8_0.gguf
S="Explain in a short paragraph why the sky is blue."; L="Write the opening two sentences of a mystery story set in a lighthouse."
C="Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring."
run q5-sky $Q5 -p "$S"; run q5-code $Q5 -p "$C"; run q5-story $Q5 -p "$L"
echo "=== OOM kills: $(dmesg | grep -ciE 'oom-kill')"
