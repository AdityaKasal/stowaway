$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
wsl -d Ubuntu-22.04 -u root -e bash -c "cd /root/moe && export PATH=/usr/local/bin:`$PATH && rsync -a /mnt/c/Users/FSociety/moe-router-study/llama.cpp/common/moe-stream.h llama.cpp/common/ && cmake --build build-dist --target llama-perplexity llama-cli llama-server -j 16 2>&1 | grep -E 'error|FAILED' | head; cp build-dist/bin/llama-perplexity build-dist/bin/llama-cli build-dist/bin/llama-server /mnt/c/Users/FSociety/moe-router-study/dist/linux-test/ && echo built"
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q38-catalog.sh
wsl --shutdown
"EXIT done"
