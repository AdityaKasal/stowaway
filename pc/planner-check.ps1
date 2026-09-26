$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
$root = "C:\Users\FSociety\moe-router-study"; Set-Location $root
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
try {
  foreach ($m in "6GB", "8GB") {
    "##### VM memory $m  $(Get-Date -Format T)"
    Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=$m`r`nprocessors=4`r`nswap=0"
    wsl --shutdown
    wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/planner-check.sh
  }
} finally {
  wsl --shutdown
  Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force; Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
}
"##### Windows, ~4.8 GB free (ram_limit), the app's planner, 35B"
$lim = Start-Process .venv\Scripts\python.exe -ArgumentList "-u", "scripts\ram_limit.py", "4.8", "900" -PassThru -WindowStyle Hidden -RedirectStandardOutput logs\pc-lim.txt
try {
  Start-Sleep 60; Get-Content logs\pc-lim.txt -Tail 1
  $env:STOWAWAY_NO_UPDATE_CHECK = "1"; $env:MOE_HOME = "C:\Users\FSociety\pc-home"
  foreach ($f in @($false, $true)) {
    $extra = @(); if ($f) { $extra = @("--fast") }
    "=== 35B story fast=$f  $(Get-Date -Format T)"
    & .venv\Scripts\python.exe moe.py run models\Qwen3.5-35B-A3B-Q5_K_M.gguf --packed E:\moe\35b-packed --bin build-dist\bin --threads 4 -n 128 @extra -p "Write the opening two sentences of a mystery story set in a lighthouse." *> logs\pc-win.txt
    Select-String -Path logs\pc-win.txt -Pattern "^machine|^small|^plan|Generation" | ForEach-Object { "  " + $_.Line.Substring(0, [Math]::Min(140, $_.Line.Length)) }
  }
} finally { Stop-Process $lim.Id -Force -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force C:\Users\FSociety\pc-home -ErrorAction SilentlyContinue }
"EXIT done"
