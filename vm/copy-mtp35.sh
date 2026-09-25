#!/bin/bash
D=/root/moe/models; t0=$(date +%s)
cp /mnt/d/moe/Qwen3.5-35B-A3B-MTP-UD-Q5_K_M.gguf $D/ && cp /mnt/d/moe/35b-mtp-experts-packed.idx /mnt/d/moe/35b-mtp-experts-packed.bin $D/
echo "copied in $(( $(date +%s) - t0 )) s"; ls -la $D/*mtp* | awk '{print $5, $9}'; df -h / | tail -1
