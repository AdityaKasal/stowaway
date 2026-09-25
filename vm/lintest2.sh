#!/bin/bash
cd /root/moe && cp /mnt/c/Users/FSociety/moe-router-study/dist/linux/* app/ && rm -f app/moe && chmod +x app/*
app/stowaway list | head -3
app/stowaway run models/Qwen3.5-35B-A3B-Q5_K_M.gguf --fast -p "Name three planets." -n 24 < /dev/null 2>&1 | grep -E "Generation|^fast|^plan|rror"
echo "slimmed 35B model file on disk: $(du -BG models/Qwen3.5-35B-A3B-Q5_K_M.gguf | cut -f1)"
