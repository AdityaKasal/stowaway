#!/bin/bash
# The release path for qwen3.8-next: catalog name, plan, list, a short run with --fast (which must be switched off)
cd /root/moe && SRC=/mnt/c/Users/FSociety/moe-router-study && cp $SRC/{moe.py,streampack.py,sparse.py,repack_experts.py,pack_dense.py} .
cp $SRC/dist/linux-test/llama-cli $SRC/dist/linux-test/llama-server app/ && chmod +x app/llama-*
H=/root/moe-home; mkdir -p $H/qwen3.8-next
for f in models/q38-iq4xs/*; do ln -sf /root/moe/$f $H/qwen3.8-next/; done
export MOE_HOME=$H STOWAWAY_NO_UPDATE_CHECK=1
python3 moe.py list 2>&1 | head -8 | cut -c1-110
python3 moe.py plan qwen3.8-next 2>&1 | cut -c1-170
python3 moe.py run qwen3.8-next --bin app --threads 4 -n 40 --draft none --fast -p "Name three rivers in Europe." < /dev/null > /tmp/c.out 2> /tmp/c.err
echo "exit $?"; grep -hE "^(fast|plan):" /tmp/c.out /tmp/c.err | cut -c1-170
grep -ohE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" /tmp/c.out /tmp/c.err
grep -A3 "Name three rivers" /tmp/c.out | tail -3 | cut -c1-200
rm -rf $H
