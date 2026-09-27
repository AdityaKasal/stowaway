while ((Get-ScheduledTask -TaskName moe-q38slim).State -eq "Running") { Start-Sleep 30 }
$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q38-slim2.sh
wsl --shutdown
"EXIT done"
