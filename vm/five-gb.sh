#!/bin/bash
# ~5 GB free (a 6 GB VM), disk at 3 GB/s: the models a typical 8 GB Windows laptop would run
cd /root/moe
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
free -m | head -2 | tail -1
L="Write the opening two sentences of a mystery story set in a lighthouse."
run() {
  echo "=== $1  $(date +%T)"; shift
  STOWAWAY_NO_UPDATE_CHECK=1 app/stowaway run "$@" --threads 4 -n 64 -p "$L" < /dev/null > /tmp/f.out 2> /tmp/f.err
  echo "  exit $?"; grep -hE "^(machine|small|plan):" /tmp/f.out /tmp/f.err | sed 's/^/  /' | cut -c1-150
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/f.out /tmp/f.err | sed 's/^/  /'
}
run qwen3.5-35b models/Qwen3.5-35B-A3B-Q5_K_M.gguf --draft none
run gpt-oss-20b models/gpt-oss/gpt-oss-20b-MXFP4.gguf
run gpt-oss-120b models/gpt-oss/gpt-oss-120b-MXFP4.gguf
