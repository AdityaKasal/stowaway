#!/bin/bash
# 8 GB VM with the disk capped at 550 MB/s (a SATA SSD): the three recommended models, one story prompt each
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux-test/llama-* app/ 2>/dev/null; chmod +x app/*
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moesata
echo "$(cat /sys/block/$DEV/dev) rbps=550000000" > /sys/fs/cgroup/moesata/io.max && echo $$ > /sys/fs/cgroup/moesata/cgroup.procs
echo "disk cap: $(cat /sys/fs/cgroup/moesata/io.max)"
L="Write the opening two sentences of a mystery story set in a lighthouse."
run() {
  echo "=== $1  $(date +%T)"; shift
  STOWAWAY_NO_UPDATE_CHECK=1 app/stowaway run "$@" --threads 4 -n 64 -p "$L" < /dev/null > /tmp/s.out 2> /tmp/s.err
  echo "  exit $?"; grep -hE "^(machine|plan):" /tmp/s.out /tmp/s.err | sed 's/^/  /' | cut -c1-150
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/s.out /tmp/s.err | sed 's/^/  /'
}
run gpt-oss-20b models/gpt-oss/gpt-oss-20b-MXFP4.gguf
run qwen3.5-35b-q5 models/Qwen3.5-35B-A3B-Q5_K_M.gguf --draft none
run gpt-oss-120b models/gpt-oss/gpt-oss-120b-MXFP4.gguf
