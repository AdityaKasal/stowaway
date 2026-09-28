$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
wsl -d Ubuntu-22.04 -u root -e bash -c "cd /root/moe && export PATH=/usr/local/bin:`$PATH && rsync -a /mnt/c/Users/FSociety/moe-router-study/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c llama.cpp/ggml/src/ggml-cpu/ && cmake --build build-dist --target llama-cli -j 16 2>&1 | grep -E 'error|FAILED' | head; mkdir -p /mnt/c/Users/FSociety/moe-router-study/dist/linux-prof && cp build-dist/bin/llama-cli /mnt/c/Users/FSociety/moe-router-study/dist/linux-prof/ && echo built"
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
try {
  Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=16GB`r`nprocessors=4`r`nswap=0"
  wsl --shutdown
  wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q38-opprof.sh
} finally {
  wsl --shutdown
  Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force; Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
}
"EXIT done"
