#!/bin/bash
# MTP head vs 0.8B helper vs nothing, in the 8 GB / 4 CPU VM with the disk capped at 3 GB/s (cgroup io.max).
# usage: test-mtp.sh <tag> <model.gguf> <packed> <cache_gb> [dense_gb]
cd /root/moe
TAG=$1 MODEL=$2 PACKED=$3 CACHE=$4 DENSE=$5
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
echo "disk cap: $(cat /sys/fs/cgroup/moe3g/io.max)"
export LLAMA_NO_MMAP_PREFETCH=1 EXPERT_CACHE_PACKED=$PACKED EXPERT_CACHE_CHUNK_KB=8192 MOE_CACHE_GB=$CACHE MOE_IO_THREADS=4 MOE_PREGATE=0
[ -n "$DENSE" ] && export EXPERT_CACHE_DENSE_GB=$DENSE EXPERT_CACHE_DENSE_PACKED=${PACKED/experts-packed/dense-packed}
BIN=llama.cpp/build/bin/llama-cli
run() {  # name prompt spec...
  local name=$1 prompt=$2; shift 2
  $BIN -m $MODEL -ngl 0 --no-repack --no-op-offload -c 4096 -b 128 -ub 128 -t 4 -tb 4 -n 96 --temp 0 --seed 1 \
       -p "$prompt" -st --simple-io --no-display-prompt --no-warmup "$@" > /tmp/$name.out 2> /tmp/$name.err
  printf "%-22s exit %s  %s\n" $name $? "$(grep -oE 'Generation: [0-9.]+ t/s' /tmp/$name.out)"
}
HELPER="-md models/Qwen3.5-0.8B-Q4_K_M.gguf -ngld 0 --spec-type draft-simple --spec-draft-n-max 12 --spec-draft-p-min 0.8"
for pn in story code; do
  [ $pn = story ] && P="Write the opening two sentences of a mystery story set in a lighthouse." \
                  || P="Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring."
  run $TAG-$pn-none "$P"
  run $TAG-$pn-helper "$P" $HELPER
  for n in 2 3 4; do run $TAG-$pn-mtp$n "$P" --spec-type draft-mtp --spec-draft-n-max $n; done
  run $TAG-$pn-mtp3-p05 "$P" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5
  ref=$(sed 's/\[ Prompt.*//' /tmp/$TAG-$pn-none.out)
  for v in helper mtp2 mtp3 mtp4 mtp3-p05; do
    [ "$(sed 's/\[ Prompt.*//' /tmp/$TAG-$pn-$v.out)" = "$ref" ] && echo "  $v: same text" || echo "  $v: DIFFERENT text"
  done
done
grep -hE "draft acceptance|accepted" /tmp/$TAG-*-mtp3.err | head -4
