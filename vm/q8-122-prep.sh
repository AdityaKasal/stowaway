#!/bin/bash
# 122B Q8: download the 4 parts into the VM, then one short run so moe packs the experts and always-needed weights.
set -e
cd /root/moe && mkdir -p models/122b-q8
HF=https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/Q8_0
for i in 1 2 3 4; do
  f=models/122b-q8/Qwen3.5-122B-A10B-Q8_0-0000$i-of-00004.gguf
  [ -f $f ] && continue
  echo "part $i start $(date +%T)"
  curl -sSL -C - --retry 30 --retry-delay 10 -o $f.part $HF/$(basename $f) && mv $f.part $f
  echo "part $i done $(date +%T): $(du -h $f | cut -f1)"
done
cp -n models/Qwen3.5-0.8B-Q4_K_M.gguf models/122b-q8/ 2>/dev/null || true
app/moe run models/122b-q8/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf -p "Hi" -n 2 --ram 5.5 --draft none < /dev/null 2>&1 \
  | grep -vE "^(llama_|load|print_info|common_|build|system_info|sampler|generate|ggml|gguf|\.)" | tail -8
ls -la models/122b-q8; df -h / | tail -1
echo "prep done $(date +%T)"
