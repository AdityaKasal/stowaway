#!/bin/bash
# Qwen3.8-Flash-Next UD-IQ4_XS speed in a memory-capped VM, disk capped at 3 GB/s (a typical laptop NVMe).
cd /root/moe
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
export STOWAWAY_NO_UPDATE_CHECK=1
M=models/q38-iq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
run() {
  echo "=== $1  $(date +%T)"; shift
  MOE_STATS=1 python3 moe.py run $M --bin app --threads 4 -n 96 --draft none "$@" < /dev/null > /tmp/i.out 2> /tmp/i.err
  echo "  exit $?, OOM kills so far: $(dmesg 2>/dev/null | grep -c 'Out of memory')"
  grep -hE "^(machine|small|plan):" /tmp/i.out /tmp/i.err | sed 's/^/  /' | cut -c1-170
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/i.out /tmp/i.err | sed 's/^/  /'
  grep -hE "expert cache:|hit rate|read .* GB" /tmp/i.err | tail -2 | sed 's/^/  /' | cut -c1-170
}
L="Write the opening two sentences of a mystery story set in a lighthouse."; S="Explain in a short paragraph why the sky is blue."
run "story" -p "$L"
run "sky" -p "$S"
[ "$(free -g | awk '/Mem:/{print $2}')" -ge 12 ] && run "story --fast" -p "$L" --fast
exit 0
