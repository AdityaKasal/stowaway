wsl -d Ubuntu-22.04 -u root -e bash -c "cd /root/moe && export PATH=/usr/local/bin:`$PATH && cmake --build build-dist --target llama-perplexity -j 16 2>&1 | tail -1 && cp build-dist/bin/llama-perplexity /mnt/c/Users/FSociety/moe-router-study/dist/linux-tools-perplexity"
wsl -d Ubuntu -u root -e bash -c "cp /mnt/c/Users/FSociety/moe-router-study/dist/linux-tools-perplexity /root/moe/app/llama-perplexity && chmod +x /root/moe/app/llama-perplexity"
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/gptoss-kld.sh
"EXIT done"
