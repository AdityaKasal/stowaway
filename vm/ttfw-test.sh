#!/bin/bash
# Time to first word, 122B Q8 on the 8 GB VM at 3 GB/s: a chat-sized question, with and without batch-aware routing
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux/* app/ && chmod +x app/*
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
M=models/122b-q8/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf
Q1="I'm planning a week-long trip to Japan in April with my parents, who don't walk much. Which two cities should we base ourselves in, and why?"
Q2="Explain the difference between a Roth IRA and a traditional IRA to someone who just got their first full-time job."
run() {  # $1 name, $2 batch bonus, $3 question
  echo "=== $1 (batch bonus $2)  $(date +%T)"
  local t0=$(date +%s.%N)
  MOE_STATS=1 MOE_BATCH_BONUS=$2 app/stowaway run $M --threads 4 -n 8 --draft none -p "$3" < /dev/null > /tmp/t.out 2> /tmp/t.err
  echo "  exit $?, total $(echo "$(date +%s.%N) - $t0" | bc | cut -c1-5) s"
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/t.out /tmp/t.err | sed 's/^/  /'
  grep -hE "batch-aware|prompt eval time" /tmp/t.err | sed 's/^/  /' | cut -c1-170
}
for q in 1 2; do eval p=\$Q$q
  for b in 0 1.0; do run "q$q" $b "$p"; done
done
