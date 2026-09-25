#!/bin/bash
# Slim mode test on the VM's 35B Q5 copy, with the new Linux build:
#  1. reference answers (packed experts, normal model file), big cache and tiny cache (tiny forces the overflow fallback)
#  2. re-pack the experts with --slim through the real app (frees the model file's own copy layer by layer)
#  3. same answers again: they must be byte-identical; disk use must drop by ~the expert size
cd /root/moe
cp /mnt/c/Users/FSociety/moe-router-study/dist/linux/* app/ && chmod +x app/*
M=models/Qwen3.5-35B-A3B-Q5_K_M.gguf; PK=models/Qwen3.5-35B-A3B-Q5_K_M-experts-packed
P="Explain what a solid state drive is, then list three things that make one faster than another."
ask() {  # $1 = cache GB, $2 = output file; same flags moe uses, greedy sampling so answers are comparable
  MOE_CACHE_GB=$1 MOE_IO_THREADS=4 MOE_PREGATE=6 EXPERT_CACHE_PACKED=$PK EXPERT_CACHE_CHUNK_KB=8192 LLAMA_NO_MMAP_PREFETCH=1 MOE_STATS=1 \
    app/llama-cli -m $M -ngl 0 --no-repack --no-op-offload -c 2048 -b 128 -ub 128 -t 8 -n 80 --no-warmup -rea off \
    --temp 0 --seed 1 -p "$P" -st --simple-io --no-display-prompt < /dev/null > $2 2> $2.err
  echo "  cache $1 GB: exit $?, $(grep -oE 'Generation: [0-9.]+ t/s' $2 | head -1), $(grep -oE '[0-9]+ experts didn.t fit' $2.err || echo 'no overflow')"
}
echo "=== before: model file $(du -BG --apparent-size $M | cut -f1) apparent, $(du -BG $M | cut -f1) on disk"
ask 6 /tmp/ref-big.txt; ask 0.3 /tmp/ref-tiny.txt
cmp -s /tmp/ref-big.txt /tmp/ref-tiny.txt && echo "  big and tiny cache answers identical" || echo "  NOTE: big vs tiny differ"
echo "=== re-pack with --slim through the app ($(date +%T))"
rm -f $PK.bin $PK.idx
df -BG / | tail -1
app/moe run $M --slim -p "Hi" -n 4 --ram 7 < /dev/null 2>&1 | grep -E "one-time|layer (0|39) done|wrote|freed|resuming|Error|error" | head -8
echo "=== after: model file $(du -BG --apparent-size $M | cut -f1) apparent, $(du -BG $M | cut -f1) on disk; packed $(du -BG $PK.bin | cut -f1)"
df -BG / | tail -1
ask 6 /tmp/slim-big.txt; ask 0.3 /tmp/slim-tiny.txt
for x in big tiny; do cmp -s /tmp/ref-$x.txt /tmp/slim-$x.txt && echo "  $x cache: IDENTICAL to before slimming" || { echo "  $x cache: DIFFERENT"; diff <(head -c 400 /tmp/ref-$x.txt) <(head -c 400 /tmp/slim-$x.txt) | head -6; }; done
head -c 300 /tmp/slim-tiny.txt; echo
echo "EXIT done"
