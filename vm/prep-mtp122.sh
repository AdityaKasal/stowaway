#!/bin/bash
# Copy the packed 122B MTP files into the VM disk.
D=/root/moe/models/122b-mtp; mkdir -p $D; t0=$(date +%s)
cp /mnt/d/moe/122b-mtp/*.gguf $D/ && cp /mnt/d/moe/122b-mtp-experts-packed.* /mnt/d/moe/122b-mtp-dense-packed.* $D/
echo "copied in $(( $(date +%s) - t0 )) s"; ls $D; df -h / | tail -1
