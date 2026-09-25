#!/bin/bash
# Copy the models into the VM's own disk (reading them through /mnt/c during the test would be unfair).
C=/mnt/c/Users/FSociety/moe-router-study/models
E=/mnt/e/moe
D=/root/moe/models
mkdir -p $D/122b
t0=$(date +%s)
cp_one() { local s=$(date +%s); cp "$1" "$2"; echo "$(basename "$1"): $(( $(stat -c %s "$1") / 1000000000 )) GB in $(( $(date +%s) - s )) s"; }
cp_one $C/Qwen3.5-0.8B-Q4_K_M.gguf $D/
[ "$1" = "small" ] && exit 0
cp_one $C/122b/dense-packed.bin $D/122b/; cp $C/122b/dense-packed.idx $D/122b/
cp $C/122b/experts-packed.idx $D/122b/; cp_one $C/122b/experts-packed.bin $D/122b/
for f in $C/122b/Qwen3.5-122B-A10B-Q5_K_M-0000*-of-00003.gguf; do cp_one $f $D/122b/; done
cp_one $C/Qwen3.5-35B-A3B-Q5_K_M.gguf $D/
cp $E/35b-packed.idx $D/; cp_one $E/35b-packed.bin $D/
echo "all copied in $(( $(date +%s) - t0 )) s"; df -h / | tail -1
