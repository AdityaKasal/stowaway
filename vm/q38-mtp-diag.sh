#!/bin/bash
# Why was MTP slower? Acceptance (log level 4) with the model's default sampling and greedy, and with pre-gating off
cd /root/moe && export STOWAWAY_NO_UPDATE_CHECK=1
H=models/q38-iq4xs/mtp/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf; M=models/q38-iq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
L="Write the opening two sentences of a mystery story set in a lighthouse."
FREE=$(awk '/MemAvailable/{printf "%.1f", $2/1048576 - 0.6}' /proc/meminfo)
run() {
  echo "=== $1  $(date +%T)"; shift
  MOE_STATS=1 python3 moe.py run $M --bin app-mtp --threads 4 -n 96 --draft none --ram $FREE -p "$L" "$@" < /dev/null > /tmp/d.out 2> /tmp/d.err
  echo "  exit $?"; grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/d.out /tmp/d.err | sed 's/^/  /'
  grep -hE "draft acceptance|acc per pos|statistics|#calls|#gen drafts|#acc drafts" /tmp/d.out /tmp/d.err | tail -6 | sed 's/^/  /' | cut -c1-170
  grep -hE "generation only: read" /tmp/d.err | sed 's/^/  /' | cut -c1-120
}
MTP="-md $H --spec-type draft-mtp --spec-draft-n-max 2 -lv 4"
STOWAWAY_LLAMA_ARGS="-lv 4" run "no MTP, default sampling"
STOWAWAY_LLAMA_ARGS="$MTP" run "MTP n=2, default sampling"
STOWAWAY_LLAMA_ARGS="$MTP --temp 0" run "MTP n=2, greedy"
STOWAWAY_LLAMA_ARGS="-lv 4 --temp 0" run "no MTP, greedy"
MOE_PREGATE=0 STOWAWAY_LLAMA_ARGS="$MTP --temp 0" run "MTP n=2, greedy, pre-gating off"
STOWAWAY_LLAMA_ARGS="-md $H --spec-type draft-mtp --spec-draft-n-max 1 -lv 4 --temp 0" run "MTP n=1, greedy"
