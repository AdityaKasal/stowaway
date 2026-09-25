#!/bin/bash
# 35B Q5 on a 4 GB VM at 3 GB/s: what actually fits? smaller context/batch, several cache/dense splits, peak memory
cd /root/moe
M=models/Qwen3.5-35B-A3B-Q5_K_M.gguf; PK=models/Qwen3.5-35B-A3B-Q5_K_M-experts-packed; DP=models/Qwen3.5-35B-A3B-Q5_K_M-dense-packed
[ -f $DP.bin ] || python3 pack_dense.py $M $DP | tail -1
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
sync; echo 3 > /proc/sys/vm/drop_caches; free -m | head -2
probe() {  # $1 name, $2 cache GB, $3 dense stream GB (0 = leave dense to the OS), $4 ctx, $5 batch
  echo "=== $1: cache $2, dense $3, ctx $4, batch $5  $(date +%T)"
  local extra=""; [ "$3" != "0" ] && extra="EXPERT_CACHE_DENSE_GB=$3 EXPERT_CACHE_DENSE_PACKED=$DP"
  env MOE_CACHE_GB=$2 MOE_IO_THREADS=4 MOE_PREGATE=$6 MOE_STATS=1 LLAMA_NO_MMAP_PREFETCH=1 EXPERT_CACHE_PACKED=$PK EXPERT_CACHE_CHUNK_KB=8192 $extra \
    app/llama-cli -m $M -ngl 0 --no-repack --no-op-offload -c $4 -b $5 -ub $5 -t 4 -tb 4 -n 64 --no-warmup -rea off \
    -p "Explain in a short paragraph why the sky is blue." -st --simple-io --no-display-prompt < /dev/null > /tmp/p.out 2> /tmp/p.err &
  local pid=$! lo=999999 hwm=0
  while kill -0 $pid 2>/dev/null; do
    a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a
    h=$(awk '/VmHWM/ {print int($2/1024)}' /proc/$pid/status 2>/dev/null); [ -n "$h" ] && hwm=$h
    sleep 0.5; done
  wait $pid; echo "  exit $?, peak process memory ${hwm} MB, lowest free ${lo} MB"
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/p.out /tmp/p.err | sed 's/^/  /'
  grep -hiE "Killed|out of memory|failed to" /tmp/p.err | head -2; dmesg | grep -ciE "oom-kill" | sed 's/^/  oom-kills so far: /'
}
probe A 0.3 0.8 2048 64 0
probe B 0.5 1.0 2048 64 0
probe C 0.8 1.0 2048 64 0
probe D 0.3 1.4 2048 64 0
probe E 0.5 0   2048 64 6
probe F 1.0 0   2048 64 6
