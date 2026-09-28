#!/bin/bash
# Per-op wall time for Qwen3.8 generation (GGML_CPU_PROFILE, thread 0's view)
cd /root/moe && export STOWAWAY_NO_UPDATE_CHECK=1 && mkdir -p app-prof && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux-prof/llama-cli app-prof/ && chmod +x app-prof/llama-cli
M=models/q38-iq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
for n in 8 200; do
  echo "=== -n $n  $(date +%T)"
  GGML_CPU_PROFILE=1 MOE_STATS=1 python3 moe.py run $M --bin app-prof --threads 4 -n $n --draft none -p "Write a short story about a lighthouse keeper." < /dev/null > /tmp/o.out 2> /tmp/o.err
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/o.out /tmp/o.err
  grep -h "cpu-profile\|generation only: read" /tmp/o.err | cut -c1-120
done
