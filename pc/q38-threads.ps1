$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
try {
  foreach ($c in @(@("4", "2 3 4"), @("8", "4 6 8"))) {
    "##### 16 GB VM, $($c[0]) CPUs"
    Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=16GB`r`nprocessors=$($c[0])`r`nswap=0"
    wsl --shutdown
    wsl -d Ubuntu -u root -e bash -c "bash /mnt/c/Users/FSociety/moe-router-study/vm/q38-threads.sh $($c[1])"
  }
} finally {
  wsl --shutdown
  Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force; Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
}
"EXIT done"
