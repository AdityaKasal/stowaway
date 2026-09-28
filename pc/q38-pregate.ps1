$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
try {
  foreach ($m in "9GB", "12GB") {
    if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game started: stopping"; break }
    "##### VM memory $m"
    Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=$m`r`nprocessors=4`r`nswap=0"
    wsl --shutdown
    wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q38-pregate.sh
  }
} finally {
  wsl --shutdown
  Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force; Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
}
"EXIT done"
