#!/bin/bash
# Is Qwen3.8 losing time to too many busy threads? Same prompt, different compute-thread counts, on this VM's CPUs.
cd /root/moe && export STOWAWAY_NO_UPDATE_CHECK=1
M=models/q38-iq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
L="Write the opening two sentences of a mystery story set in a lighthouse."
echo "CPUs in the VM: $(nproc)"
for t in "$@"; do
  echo "=== --threads $t  $(date +%T)"
  MOE_STATS=1 python3 moe.py run $M --bin app --threads $t -n 96 --draft none -p "$L" < /dev/null > /tmp/t.out 2> /tmp/t.err
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/t.out /tmp/t.err | sed 's/^/  /'
  grep -hE "generation only: read" /tmp/t.err | sed 's/^/  /' | cut -c1-110
done
