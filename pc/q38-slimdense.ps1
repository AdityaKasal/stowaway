# after moe-q38next: the Q5_1 experiment (quality in the default VM, then speed at 8 GB)
while ((Get-ScheduledTask -TaskName moe-q38next).State -eq "Running") { Start-Sleep 30 }
$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
"##### quality  $(Get-Date -Format T)"
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q38-slimdense.sh
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
try {
  foreach ($m in "8GB", "6GB", "16GB") {
    if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game started: stopping"; break }
    "##### speed with Q5_1 file 3, VM memory $m"
    Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=$m`r`nprocessors=4`r`nswap=0"
    wsl --shutdown
    wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q38-speed.sh
  }
} finally {
  wsl --shutdown
  Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force; Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
}
"EXIT done"
