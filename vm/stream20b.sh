#!/bin/bash
# direct download of gpt-oss-20b in the VM (Linux): must equal the packed file made by repack_experts --slim earlier
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/{moe.py,streampack.py,sparse.py,repack_experts.py} .
D=/root/moe/stream-test; rm -rf $D; mkdir -p $D
t0=$(date +%s)
python3 -c "
import moe, streampack
e = moe.CATALOG['gpt-oss-20b']
exp = [moe.hf_checksum(e['repo'], f) for f in e['files']]
streampack.fetch_packed(e['repo'], e['files'], '$D', '$D/gpt-oss-20b-MXFP4-experts-packed', exp, 'stowaway-test')
" 2>&1 | tr '\r' '\n' | grep -vE "GB  \(" | tail -3
echo "took $(( $(date +%s) - t0 )) s"
R=models/gpt-oss
cmp $D/gpt-oss-20b-MXFP4-experts-packed.bin $R/gpt-oss-20b-MXFP4-experts-packed.bin && echo "packed .bin: IDENTICAL to repack --slim"
cmp $D/gpt-oss-20b-MXFP4-experts-packed.idx $R/gpt-oss-20b-MXFP4-experts-packed.idx && echo "index: IDENTICAL"
cmp $D/gpt-oss-20b-MXFP4.gguf $R/gpt-oss-20b-MXFP4.gguf && echo "model file: IDENTICAL to the slimmed original"
echo "disk use: $(du -sh $D | cut -f1) (model file $(du -BM $D/gpt-oss-20b-MXFP4.gguf | cut -f1) on disk)"
sha256sum $D/gpt-oss-20b-MXFP4-experts-packed.bin | cut -c1-16
