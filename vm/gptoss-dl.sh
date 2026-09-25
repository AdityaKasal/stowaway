#!/bin/bash
# download gpt-oss-20b and -120b (MXFP4, ggml-org) into the VM
cd /root/moe && mkdir -p models/gpt-oss
for m in gpt-oss-20b gpt-oss-120b; do
  f=models/gpt-oss/$m-MXFP4.gguf; [ -f $f ] && continue
  echo "$m start $(date +%T)"
  curl -sSL -C - --retry 30 --retry-delay 10 -o $f.part https://huggingface.co/ggml-org/$m-GGUF/resolve/main/$m-MXFP4.gguf && mv $f.part $f
  echo "$m done $(date +%T): $(du -h $f | cut -f1)"
done
df -h / | tail -1
