#!/bin/bash
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/repack_experts.py /mnt/c/Users/FSociety/moe-router-study/sparse.py .
M=models/Qwen3.5-35B-A3B-Q8_0.gguf
echo "before: $(du -BG $M | cut -f1)"; python3 -c "import repack_experts as r; r.slim_after('$M', 'models/Qwen3.5-35B-A3B-Q8_0-experts-packed')"; echo "after: $(du -BG $M | cut -f1)"
du -sh models; df -h / | tail -1
