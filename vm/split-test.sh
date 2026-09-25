#!/bin/bash
# 122B Q8 on the 8 GB VM at 3 GB/s: how to split memory between the expert cache and the streamed always-needed weights
cd /root/moe
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
echo "disk cap: $(cat /sys/fs/cgroup/moe3g/io.max)"; free -m | head -2
D=models/122b-q8; M=$D/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf
low_mem() { local lo=999999; while kill -0 $1 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done; echo "  lowest free memory: ${lo} MB"; }
run() {  # $1 cache GB, $2 dense budget GB, $3 name, $4 prompt
  echo "=== cache $1 GB + dense $2 GB  [$3]  $(date +%T)"
  MOE_CACHE_GB=$1 EXPERT_CACHE_DENSE_GB=$2 MOE_IO_THREADS=4 MOE_PREGATE=0 MOE_STATS=1 LLAMA_NO_MMAP_PREFETCH=1 \
  EXPERT_CACHE_PACKED=$D/Qwen3.5-122B-A10B-Q8_0-experts-packed EXPERT_CACHE_DENSE_PACKED=$D/Qwen3.5-122B-A10B-Q8_0-dense-packed \
  EXPERT_CACHE_CHUNK_KB=8192 CUDA_VISIBLE_DEVICES=-1 \
    app/llama-cli -m $M -ngl 0 --no-repack --no-op-offload -c 4096 -b 128 -ub 128 -t 4 -tb 4 -n 64 --no-warmup -rea off \
    -p "$4" -st --simple-io --no-display-prompt < /dev/null > /tmp/split.out 2> /tmp/split.err &
  local pid=$!; low_mem $pid & local m=$!; wait $pid; echo "  exit $?"; wait $m
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/split.out /tmp/split.err | sed 's/^/  /'
  grep -hE "generation only|dense streaming" /tmp/split.err | sed 's/^/  /' | cut -c1-160
  grep -hiE "Killed|out of memory|failed" /tmp/split.err | head -2
}
P1="Explain in a short paragraph why the sky is blue."
P3="Write the opening two sentences of a mystery story set in a lighthouse."
for split in "0.5 4.7" "0.3 4.9" "1.0 4.2" "1.5 3.7" "2.2 3.0" "1.2 4.7" "0.5 5.4"; do
  set -- $split; run $1 $2 sky "$P1"; run $1 $2 story "$P3"
done
echo "=== OOM kills: $(dmesg 2>/dev/null | grep -ciE 'out of memory|oom-kill')"
