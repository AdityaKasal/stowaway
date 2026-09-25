#!/bin/bash
# 122B Q8 on a 16 GB / 4 CPU / no-swap VM, disk capped at 3 GB/s, with the released app: plain, --fast, helper
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux/* app/ 2>/dev/null; chmod +x app/*
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
free -m | head -2
M=models/122b-q8/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf
run() {
  echo "=== $1  $(date +%T)"; shift
  MOE_STATS=1 app/stowaway run $M --threads 4 -n 96 "$@" < /dev/null > /tmp/r16.out 2> /tmp/r16.err &
  local pid=$! lo=999999; while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done
  wait $pid; echo "  exit $?, lowest free ${lo} MB"
  grep -hE "^(machine|plan|guessing|fast):" /tmp/r16.out /tmp/r16.err | sed 's/^/  /' | cut -c1-170
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/r16.out /tmp/r16.err | sed 's/^/  /'
  grep -hE "routing \(" /tmp/r16.err | sed 's/^/  /' | cut -c1-150
}
S="Explain in a short paragraph why the sky is blue."; L="Write the opening two sentences of a mystery story set in a lighthouse."
run plain-sky --draft none -p "$S"; run plain-story --draft none -p "$L"
run fast-sky --draft none --fast -p "$S"; run fast-story --draft none --fast -p "$L"
run helper-story -p "$L"
echo "=== OOM kills: $(dmesg | grep -ciE 'oom-kill')"
