#!/bin/zsh
# Compare ways of running a model bigger than RAM. Records tok/s and SSD read rate for each.
#
#   ./bench.sh cpu           no GPU at all; every weight read from the file on demand (minimal resources)
#   ./bench.sh naive         everything on the GPU (the normal way; needs the whole model in memory)
#   ./bench.sh stream        attention on GPU, experts left in the file and paged in from SSD by macOS
#   ./bench.sh prefetch      same as stream, but we tell macOS which experts to read as soon as the router picks them
#
# extra args are passed through, e.g. ./bench.sh stream -n 64
set -u
cd "${0:A:h}"

MODEL=models/Qwen3.5-35B-A3B-Q5_K_M.gguf
MODE=${1:-stream}; shift || true
OUT=results/$MODE-$(date +%H%M%S)
mkdir -p $OUT

case $MODE in
  naive)    FLAGS=(-ngl 99) ;;
  cpu)      FLAGS=(-ngl 0) ;;
  cpu-prefetch) FLAGS=(-ngl 0 --prefetch) ;;
  stream)   FLAGS=(-ngl 99 --n-cpu-moe 99) ;;
  prefetch) FLAGS=(-ngl 99 --n-cpu-moe 99 --prefetch) ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

# SSD reads, sampled every second while the model runs
iostat -d -w 1 disk0 > $OUT/iostat.txt &
IOSTAT=$!

/usr/bin/time -l expert-logger/expert-logger -m $MODEL --prompts bench-prompts.tsv --out-dir $OUT --no-log \
  -c 4096 -b 512 -fa on --no-repack -n 128 --seed 1 $FLAGS "$@" 2> $OUT/log.txt
STATUS=$?
kill $IOSTAT

echo "== $MODE (exit $STATUS)"
grep -E "^\[|prefetch:" $OUT/log.txt
grep -E "maximum resident|page faults|real" $OUT/log.txt | head -4
# average MB/s over the samples where the disk was busy
awk 'NR>2 && $3>1 {s+=$3; n++} END {if (n) printf "SSD reads while busy: %.0f MB/s average over %d s\n", s/n, n}' $OUT/iostat.txt
echo "full log: $OUT/log.txt"
