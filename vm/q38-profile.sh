#!/bin/bash
# Where do Qwen3.8's ~0.19 s per word of computing go? Poor man's profiler: while the model generates, grab every
# thread's stack 40 times (gdb), then count the innermost llama.cpp/ggml function per compute thread.
cd /root/moe && export STOWAWAY_NO_UPDATE_CHECK=1 DEBIAN_FRONTEND=noninteractive
command -v gdb > /dev/null || apt-get install -y -qq gdb > /dev/null 2>&1
M=models/q38-iq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
python3 moe.py run $M --bin app --threads 4 -n 400 --draft none -p "Write a long story about a lighthouse keeper." < /dev/null > /tmp/p.out 2> /tmp/p.err &
for i in $(seq 1 120); do P=$(pgrep -f "app/llama-cli" | head -1); [ -n "$P" ] && grep -q "Generation\|lighthouse" /tmp/p.out 2>/dev/null && break; sleep 1; done
sleep 25  # past the prompt, into generation
P=$(pgrep -f "app/llama-cli" | head -1); echo "profiling pid $P"
rm -f /tmp/stacks.txt
for i in $(seq 1 40); do
  gdb -p $P -batch -ex "set pagination off" -ex "thread apply all bt 12" 2>/dev/null | grep -E "^#|^Thread" >> /tmp/stacks.txt
  sleep 0.7
done
wait
grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/p.out /tmp/p.err
python3 - <<'PY'
import re, collections
frames = open('/tmp/stacks.txt').read().split('\nThread ')
top = collections.Counter(); ops = collections.Counter(); n = 0
for t in frames:
    fs = re.findall(r'^#\d+\s+(?:0x[0-9a-f]+ in )?([\w:~<>]+)', t, re.M)
    if not fs or not any('ggml' in f for f in fs):
        continue
    if any(w in fs[0] for w in ('futex', 'nanosleep', 'poll', 'epoll', 'pthread_cond', 'sched_yield', '__GI___')) and 'ggml' not in fs[0]:
        top['(waiting) ' + next((f for f in fs if 'ggml' in f or 'moe' in f or 'expert' in f), fs[0])] += 1
    else:
        top[fs[0]] += 1
    op = next((f for f in fs if f.startswith('ggml_compute_forward_')), None)
    ops[op or '(no op)'] += 1
    n += 1
print(f"{n} busy-thread samples")
print("innermost function:")
for k, v in top.most_common(18): print(f"  {100*v/n:5.1f}%  {k}")
print("ggml op on the stack:")
for k, v in ops.most_common(15): print(f"  {100*v/n:5.1f}%  {k}")
PY
