#!/bin/bash
cd /root/moe
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
export STOWAWAY_NO_UPDATE_CHECK=1
run() {
  echo "=== $1  $(date +%T)"; local m=$2; shift 2
  python3 moe.py run $m --bin app --threads 4 -n 96 --draft none "$@" < /dev/null > /tmp/i.out 2> /tmp/i.err
  echo "  exit $?"; grep -hE "^(machine|small|plan):" /tmp/i.out /tmp/i.err | sed 's/^/  /' | cut -c1-150
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/i.out /tmp/i.err | sed 's/^/  /'
}
L="Write the opening two sentences of a mystery story set in a lighthouse."; S="Explain in a short paragraph why the sky is blue."
run "122B UD-IQ4_XS story" models/122b-iq4xs/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf -p "$L"
run "122B UD-IQ4_XS sky" models/122b-iq4xs/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf -p "$S"
run "122B Q8 story (for comparison)" models/122b-q8/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf -p "$L"
