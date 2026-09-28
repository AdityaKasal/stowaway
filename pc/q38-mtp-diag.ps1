while ((Get-ScheduledTask -TaskName moe-q38mtp).State -eq "Running") { Start-Sleep 30 }
$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
try {
  "##### VM memory 16GB"
  Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=16GB`r`nprocessors=4`r`nswap=0"
  wsl --shutdown
  wsl -d Ubuntu -u root -e bash -c "cp /mnt/c/Users/FSociety/moe-router-study/moe.py /root/moe/ && bash /mnt/c/Users/FSociety/moe-router-study/vm/q38-mtp-diag.sh"
} finally {
  wsl --shutdown
  Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force; Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
}
"EXIT done"
