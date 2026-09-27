$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
"##### build (Linux, with the lookup-table fix)  $(Get-Date -Format T)"
wsl -d Ubuntu-22.04 -u root -e bash -c "nice -n 10 bash /mnt/c/Users/FSociety/moe-router-study/vm/build-dist.sh 2>&1 | tail -4"
"##### Qwen3.8-Flash-Next  $(Get-Date -Format T)"
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q38.sh
"EXIT done"
