#!/bin/bash
# Qwen3.8-Flash-Next (qwen4exp: 125B, 10 of 512 experts, plus a 51B n-gram table that is read a few rows per word):
# download UD-IQ4_XS straight into the packed layout, then a first run with the full VM.
cd /root/moe && SRC=/mnt/c/Users/FSociety/moe-router-study
cp $SRC/{moe.py,streampack.py,sparse.py,repack_experts.py,pack_dense.py} .
cp $SRC/dist/linux/llama-cli $SRC/dist/linux/llama-server app/ && chmod +x app/llama-*
export STOWAWAY_NO_UPDATE_CHECK=1
D=/root/moe/models/q38-iq4xs; mkdir -p $D
# the download's page cache is capped, so the VM does not grow to 15 GB of the PC's RAM while it runs
mkdir -p /sys/fs/cgroup/dl && echo 3G > /sys/fs/cgroup/dl/memory.max
( echo $BASHPID > /sys/fs/cgroup/dl/cgroup.procs; exec python3 - ) <<'PY'
import moe, streampack
repo = "unsloth/Qwen3.8-Flash-Next-GGUF"
files = [f"UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
exp = [moe.hf_checksum(repo, f) for f in files]
d = "/root/moe/models/q38-iq4xs"
streampack.fetch_packed(repo, files, d, d + "/Qwen3.8-Flash-Next-UD-IQ4_XS-experts-packed", exp, "stowaway-research")
PY
echo "download exit $? $(date +%T)"; du -sh $D; ls -la $D
games=$(powershell.exe -NoProfile -Command 'Get-Process | Where-Object { $_.ProcessName -match "Lunar|javaw|Minecraft|Roblox|Spider|Valorant" } | Measure-Object | Select -Expand Count' 2>/dev/null | tr -d '\r')
[ "${games:-0}" != "0" ] && { echo "a game is running: skipping the test run"; exit 0; }
M=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
run() {
  echo "=== $1  $(date +%T)"; shift
  MOE_STATS=1 timeout 1800 python3 moe.py run $M --bin app --threads 4 -n 128 --draft none "$@" < /dev/null > /tmp/q.out 2> /tmp/q.err
  echo "  exit $?"; grep -hE "^(model|machine|small|plan|fast):" /tmp/q.out /tmp/q.err | sed 's/^/  /' | cut -c1-220
  grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/q.out /tmp/q.err | sed 's/^/  /'
  echo "  --- answer:"; grep -vE "^(model|machine|small|plan|fast|load|build|main|llama|common|print_info|system|sampler|generate|\[)" /tmp/q.out | tail -12 | cut -c1-200 | sed 's/^/  | /'
  grep -hiE "error|failed|abort|assert" /tmp/q.err | head -5 | sed 's/^/  ! /'
  cp /tmp/q.err $SRC/logs/q38-$(echo $1 | tr ' ' '-').err
}
run "sky" -p "Explain in a short paragraph why the sky is blue."
run "story" -p "Write the opening two sentences of a mystery story set in a lighthouse."
echo EXIT done
