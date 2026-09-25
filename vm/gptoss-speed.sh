#!/bin/bash
# gpt-oss speed on the capped-memory VM at 3 GB/s with the app. $1 = model name (20b/120b), rest = runs to do
cd /root/moe
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
free -m | head -2 | tail -1
M=models/gpt-oss/gpt-oss-$1-MXFP4.gguf; shift
run() {
  echo "=== $1  $(date +%T)"; local p=$2; shift 2
  env STOWAWAY_NO_UPDATE_CHECK=1 MOE_STATS=1 app/stowaway run $M --threads 4 -n 128 "$@" -p "$p" < /dev/null > /tmp/g.out 2> /tmp/g.err &
  local pid=$! lo=999999; while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done
  wait $pid; echo "  exit $?, lowest free ${lo} MB"
  grep -hE "^(plan|small|fast):" /tmp/g.out /tmp/g.err | sed 's/^/  /' | cut -c1-150
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/g.out /tmp/g.err | sed 's/^/  /'
  grep -hE "routing \(|can't run|rror" /tmp/g.out /tmp/g.err | sed 's/^/  /' | cut -c1-140 | head -3
}
S="Explain in a short paragraph why the sky is blue."; L="Write the opening two sentences of a mystery story set in a lighthouse."
C="Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring."
for mode in "$@"; do
  case $mode in
    plain) run "plain sky" "$S"; run "plain code" "$C"; run "plain story" "$L";;
    fast)  run "fast sky" "$S" --fast; run "fast code" "$C" --fast; run "fast story" "$L" --fast;;
  esac
done
echo "OOM kills: $(dmesg | grep -ciE 'oom-kill')"
