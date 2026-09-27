# after the download + first run (task moe-q38): speed at 6/8/16 GB, then the quality cost of --fast
while ((Get-ScheduledTask -TaskName moe-q38).State -eq "Running") { Start-Sleep 30 }
if (-not (Select-String -Path C:\Users\FSociety\moe-router-study\logs\q38.log -Pattern "=== story" -Quiet)) { "first run did not happen: stopping"; "EXIT done"; exit 0 }
& C:\Users\FSociety\moe-router-study\scripts\q38-speed.ps1
$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: skipping the quality test"; "EXIT done"; exit 0 }
"##### quality of --fast (default VM)  $(Get-Date -Format T)"
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q38-kld.sh
wsl --shutdown
"EXIT done"
