#!/bin/bash
# Qwen3.8 with and without its MTP head (shared-Q8_0, --spec-draft-n-max 2), drive capped at 3 GB/s, 4 threads
cd /root/moe && SRC=/mnt/c/Users/FSociety/moe-router-study && cp $SRC/moe.py . && mkdir -p app-mtp && cp $SRC/dist/linux-mtp/llama-* app-mtp/ && chmod +x app-mtp/*
H=models/q38-iq4xs/mtp; F=mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf; mkdir -p $H
if [ ! -f $H/$F ]; then
  curl -sL -o $H/$F.part "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/$F" && mv $H/$F.part $H/$F
  python3 -c "
import moe, sys; from pathlib import Path
ok, why = moe.verify_download(Path('$H/$F'), moe.hf_checksum('unsloth/Qwen3.8-Flash-Next-GGUF', 'MTP/$F'))
print('head download ok' if ok else 'head download BAD: ' + why); sys.exit(0 if ok else 1)" || exit 1
fi
DEV=$(lsblk -no PKNAME $(findmnt -no SOURCE /) 2>/dev/null); [ -z "$DEV" ] && DEV=$(basename $(findmnt -no SOURCE /))
echo "+io" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null; mkdir -p /sys/fs/cgroup/moe3g
echo "$(cat /sys/block/$DEV/dev) rbps=3000000000" > /sys/fs/cgroup/moe3g/io.max && echo $$ > /sys/fs/cgroup/moe3g/cgroup.procs
export STOWAWAY_NO_UPDATE_CHECK=1
M=models/q38-iq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
run() {
  echo "=== $1  $(date +%T)"; shift
  MOE_STATS=1 python3 moe.py run $M --bin app-mtp --threads 4 -n 96 --draft none "$@" < /dev/null > /tmp/m.out 2> /tmp/m.err
  echo "  exit $?, OOM kills so far: $(dmesg 2>/dev/null | grep -c 'Out of memory')"
  grep -hE "^(machine|plan):" /tmp/m.out /tmp/m.err | sed 's/^/  /' | cut -c1-150
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/m.out /tmp/m.err | sed 's/^/  /'
  grep -hiE "draft acceptance|accepted|n_draft" /tmp/m.out /tmp/m.err | tail -2 | sed 's/^/  /' | cut -c1-150
  grep -hiE "error|failed" /tmp/m.err | grep -v "measure the memory\|borrow" | head -3 | sed 's/^/  ! /' | cut -c1-150
  tail -c 300 /tmp/m.out | tr '\n' ' ' | sed 's/^/  | /'; echo
}
L="Write the opening two sentences of a mystery story set in a lighthouse."; S="Explain in a short paragraph why the sky is blue."
FREE=$(awk '/MemAvailable/{printf "%.1f", $2/1048576 - 0.6}' /proc/meminfo)
MTP="-md $H/$F --spec-type draft-mtp --spec-draft-n-max 2"
run "story" -p "$L"
STOWAWAY_LLAMA_ARGS="$MTP" run "story + MTP (planned for $FREE GB)" -p "$L" --ram $FREE
run "sky" -p "$S"
STOWAWAY_LLAMA_ARGS="$MTP" run "sky + MTP (planned for $FREE GB)" -p "$S" --ram $FREE
