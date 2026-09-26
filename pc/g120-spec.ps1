# gpt-oss-120b: bigger caches + EAGLE3 at ~5 GB free (6 GB VM) and 8 GB; restores .wslconfig
$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
wsl -d Ubuntu -u root -e bash -c "cp /mnt/c/Users/FSociety/moe-router-study/dist/linux-test/llama-* /root/moe/app/ 2>/dev/null; ls -la /root/moe/app | head -8"
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
try {
  foreach ($m in "6GB", "8GB") {
    "##### VM memory $m  $(Get-Date -Format T)"
    Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=$m`r`nprocessors=4`r`nswap=0"
    wsl --shutdown
    wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/g120-spec.sh $m
  }
} finally {
  wsl --shutdown
  Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force; Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
  "wslconfig restored: " + ((Get-Content $env:USERPROFILE\.wslconfig) -join " / ")
}
"EXIT done"
