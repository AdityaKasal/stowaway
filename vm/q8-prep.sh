#!/bin/bash
# Q8 test, step 1 (normal VM, no limits): get the 35B Q8, copy the Q5 for comparison, set up both with the released moe.
set -e
cd /root/moe && mkdir -p models app
cp /mnt/c/Users/FSociety/moe-router-study/dist/linux/* app/ && chmod +x app/*
Q8=models/Qwen3.5-35B-A3B-Q8_0.gguf
if [ ! -f $Q8 ]; then
  echo "download start $(date +%T)"
  curl -sSL -C - --retry 20 --retry-delay 5 -o $Q8.part https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q8_0.gguf
  mv $Q8.part $Q8; echo "download done $(date +%T): $(du -h $Q8 | cut -f1)"
fi
[ -f models/Qwen3.5-35B-A3B-Q5_K_M.gguf ] || cp /mnt/c/Users/FSociety/moe-router-study/models/Qwen3.5-35B-A3B-Q5_K_M.gguf models/
[ -f models/Qwen3.5-35B-A3B-Q5_K_M-experts-packed.bin ] || { cp /mnt/e/moe/35b-packed.bin models/Qwen3.5-35B-A3B-Q5_K_M-experts-packed.bin; cp /mnt/e/moe/35b-packed.idx models/Qwen3.5-35B-A3B-Q5_K_M-experts-packed.idx; }
[ -f models/Qwen3.5-0.8B-Q4_K_M.gguf ] || cp /mnt/c/Users/FSociety/moe-router-study/models/Qwen3.5-0.8B-Q4_K_M.gguf models/
echo "copies done $(date +%T)"
# first run packs the Q8 experts (and the always-needed weights if they'd be streamed)
app/moe run $Q8 -p "Hi" -n 4 --ram 5.5 < /dev/null 2>&1 | grep -vE "^(llama_|load|print_info|common_|build|system_info|sampler|generate|ggml|gguf|\.)" | tail -8
ls -la models/; df -h / | tail -1
echo "prep done $(date +%T)"
