#!/bin/bash
# Qwen3.6-35B end to end through the app's direct download (VM, normal memory), then a short answer
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/{moe.py,streampack.py,sparse.py,repack_experts.py,pack_dense.py} .
export STOWAWAY_NO_UPDATE_CHECK=1 MOE_HOME=/root/moe/home36
t0=$(date +%s)
python3 moe.py run qwen3.6-35b --yes --bin app --threads 8 -n 96 -p "In three short bullet points, what should I pack for a weekend hiking trip?" < /dev/null > /tmp/q36.out 2>&1
echo "exit $?, took $(( $(date +%s) - t0 )) s"
grep -E "downloading straight|done \(|one-time|checking|^plan|Generation" /tmp/q36.out | tr '\r' '\n' | grep -vE "GB  \(" | cut -c1-140
grep -A8 "^> In three" /tmp/q36.out | head -10
du -sh $MOE_HOME/qwen3.6-35b/*
