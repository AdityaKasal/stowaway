#!/bin/bash
# the app's new planner, end to end: gpt-oss-120b and Qwen3.6-35B at this VM's memory, 3 GB/s
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/{moe.py,streampack.py,sparse.py,repack_experts.py,pack_dense.py} .
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
export STOWAWAY_NO_UPDATE_CHECK=1 MOE_HOME=/root/moe/home36
run() {
  echo "=== $1  $(date +%T)"; local m=$2; shift 2
  python3 moe.py run $m --bin app --threads 4 -n 128 "$@" < /dev/null > /tmp/pc.out 2> /tmp/pc.err &
  local pid=$! lo=999999; while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); [ $a -lt $lo ] && lo=$a; sleep 0.5; done
  wait $pid; echo "  exit $?, lowest free ${lo} MB"
  grep -hE "^(machine|small|plan|fast):" /tmp/pc.out /tmp/pc.err | sed 's/^/  /' | cut -c1-150
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/pc.out /tmp/pc.err | sed 's/^/  /'
}
S="Explain in a short paragraph why the sky is blue."; L="Write the opening two sentences of a mystery story set in a lighthouse."
G=models/gpt-oss/gpt-oss-120b-MXFP4.gguf; Q=home36/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
run "gpt-oss-120b sky" $G -p "$S"; run "gpt-oss-120b story" $G -p "$L"
run "qwen3.6-35b sky" $Q --draft none -p "$S"; run "qwen3.6-35b story" $Q --draft none -p "$L"
run "qwen3.6-35b story --fast" $Q --draft none --fast -p "$L"
echo "OOM kills: $(dmesg | grep -ciE 'oom-kill')"
