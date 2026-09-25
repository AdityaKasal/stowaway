#!/bin/bash
# (re)pack the 122B Q8 experts with the fixed repack_experts.py, slim: frees the originals' copy layer by layer
set -e
cd /root/moe
cp /mnt/c/Users/FSociety/moe-router-study/repack_experts.py /mnt/c/Users/FSociety/moe-router-study/sparse.py .
D=models/122b-q8; M=$D/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf; PK=$D/Qwen3.5-122B-A10B-Q8_0-experts-packed
echo "before: gguf $(du -BG -c $D/*.gguf | tail -1 | cut -f1) on disk"; df -BG / | tail -1
python3 repack_experts.py $M $PK --slim 2>&1 | grep -vE "layer +[0-9]+ done" 
echo "after: gguf $(du -BG -c $D/*.gguf | tail -1 | cut -f1) on disk, packed $(du -BG $PK.bin | cut -f1)"; df -BG / | tail -1
ls $PK.idx && echo SLIM-OK
