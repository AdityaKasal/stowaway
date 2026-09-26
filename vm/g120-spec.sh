#!/bin/bash
# gpt-oss-120b at the VM's memory size, 3 GB/s: bigger expert caches and the EAGLE3 guesser. $1 = label
cd /root/moe
E=models/gpt-oss/eagle3-gpt-oss-120b-Q8_0.gguf
[ -f $E ] || curl -sSL -o $E https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/eagle3-gpt-oss-120b-Q8_0.gguf
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
free -m | head -2 | tail -1
M=models/gpt-oss/gpt-oss-120b-MXFP4.gguf; PK=models/gpt-oss/gpt-oss-120b-MXFP4-experts-packed
P="Write a short paragraph explaining how a refrigerator keeps food cold."
KW='{"reasoning_effort": "low"}'
run() {  # $1 name, $2 cache GB, rest: extra args
  local name=$1 cache=$2; shift 2
  echo "=== $1 $name cache $cache  $(date +%T)"
  MOE_CACHE_GB=$cache MOE_IO_THREADS=4 MOE_PREGATE=6 MOE_STATS=1 EXPERT_CACHE_PACKED=$PK EXPERT_CACHE_CHUNK_KB=8192 LLAMA_NO_MMAP_PREFETCH=1 \
    app/llama-cli -m $M -ngl 0 --no-repack --no-op-offload -c 2048 -b 64 -ub 64 -t 4 -tb 4 -n 128 --no-warmup -rea off \
    --chat-template-kwargs "$KW" -p "$P" -st --simple-io --no-display-prompt "$@" < /dev/null > /tmp/g.out 2> /tmp/g.err &
  local pid=$! lo=999999; while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done
  wait $pid; echo "  exit $?, lowest free ${lo} MB"
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/g.out /tmp/g.err | sed 's/^/  /'
  grep -hE "generation only: read|draft acceptance|accept" /tmp/g.err | sed 's/^/  /' | cut -c1-140 | head -3
}
FREE=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
if [ $FREE -lt 6000 ]; then CACHES="0.9 1.8 2.5"; else CACHES="3.0 4.0 4.8"; fi
for c in $CACHES; do run plain $c; done
for c in $CACHES; do run eagle3-n4 $c --spec-type draft-eagle3 -md $E --spec-draft-n-max 4; done
C=$(echo $CACHES | awk '{print $2}')
run eagle3-n8 $C --spec-type draft-eagle3 -md $E --spec-draft-n-max 8
echo "OOM kills: $(dmesg | grep -ciE 'oom-kill')"
