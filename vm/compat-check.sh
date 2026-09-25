#!/bin/bash
cd /root/moe && mkdir -p app/compat && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux/compat/* app/compat/ && cp /mnt/c/Users/FSociety/moe-router-study/moe.py . && chmod +x app/compat/*
echo "VEX-encoded (AVX-family) instructions in compat llama-cli: $(objdump -d --no-show-raw-insn app/compat/llama-cli | grep -cE $'\tv(fmadd|fmsub|cvtph2ps|cvtps2ph|broadcast|perm|insert|extract|[a-z]+ps|[a-z]+pd|p[a-z]+|mov[a-z]*)\\b')"
echo "BMI2 instructions (shlx/sarx/pdep/pext/bzhi/mulx): $(objdump -d --no-show-raw-insn app/compat/llama-cli | grep -cE $'\t(shlx|shrx|sarx|pdep|pext|bzhi|mulx|rorx) ')"
for mode in normal compat; do
  [ $mode = compat ] && export STOWAWAY_COMPAT=1
  echo "=== $mode"
  python3 moe.py run models/Qwen3.5-35B-A3B-Q5_K_M.gguf --bin app --threads 8 -n 48 -p "Name three rivers in Africa." < /dev/null 2>&1 | grep -E "^cpu|Generation"
done
