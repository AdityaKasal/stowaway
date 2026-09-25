#!/bin/bash
# gpt-oss-20b: same answer with plain llama.cpp vs stowaway (packed + slimmed)? Greedy sampling so it's comparable.
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/{repack_experts.py,sparse.py,moe.py,pack_dense.py} . && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux/* app/ && chmod +x app/*
M=models/gpt-oss/gpt-oss-20b-MXFP4.gguf; PK=models/gpt-oss/gpt-oss-20b-MXFP4-experts-packed
P="Explain in two sentences why the sky is blue."
ARGS="-m $M -ngl 0 --no-repack -c 2048 -t 8 -n 64 --no-warmup -rea off --temp 0 --seed 1 -st --simple-io --no-display-prompt"
KW='{"reasoning_effort": "low"}'
app/llama-cli $ARGS --chat-template-kwargs "$KW" -p "$P" < /dev/null 2>/dev/null | grep -vE "t/s|Loading|Exiting|^[^a-zA-Z0-9]*$" > /tmp/ref.txt
echo "plain llama.cpp answer: $(wc -c < /tmp/ref.txt) bytes"; head -c 300 /tmp/ref.txt; echo
rm -f $PK.bin $PK.idx $PK.progress
echo "before slim: $(du -BM $M | cut -f1) on disk"
python3 repack_experts.py $M $PK --slim | grep -vE "layer +[0-9]+ done"
echo "after slim: $(du -BM $M | cut -f1) on disk, packed $(du -BM $PK.bin | cut -f1)"
for gb in 4 0.3; do
  MOE_CACHE_GB=$gb MOE_IO_THREADS=4 MOE_PREGATE=6 EXPERT_CACHE_PACKED=$PK EXPERT_CACHE_CHUNK_KB=8192 LLAMA_NO_MMAP_PREFETCH=1 MOE_STATS=1 \
    app/llama-cli $ARGS --no-op-offload --chat-template-kwargs "$KW" -p "$P" < /dev/null 2>/tmp/s.err | grep -vE "t/s|Loading|Exiting|^[^a-zA-Z0-9]*$" > /tmp/slim.txt
  cmp -s /tmp/ref.txt /tmp/slim.txt && echo "cache $gb GB: IDENTICAL to plain llama.cpp" || { echo "cache $gb GB: DIFFERENT"; diff /tmp/ref.txt /tmp/slim.txt | head -6; }
  grep -hE "didn't fit|already in RAM" /tmp/s.err | cut -c1-120
done
