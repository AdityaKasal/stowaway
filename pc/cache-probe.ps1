# How big can the expert cache safely get on an 8 GB Windows laptop (~4.8 GB free)? CPU-only release engine, 35B Q5.
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }
$env:PATH = "C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
$root = "C:\Users\FSociety\moe-router-study"; Set-Location $root
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
cmake --build build-dist --target llama-cli -j 8 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 5
$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
$story = "Write the opening two sentences of a mystery story set in a lighthouse."
$common = @("-m", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "-ngl", "0", "--no-repack", "--no-op-offload", "-c", "2048", "-b", "64", "-ub", "64",
            "-t", "4", "-tb", "4", "-n", "64", "--no-warmup", "-rea", "off", "-p", "`"$story`"", "-st", "--simple-io", "--no-display-prompt")
$env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:MOE_IO_THREADS = "4"; $env:MOE_PREGATE = "6"; $env:MOE_STATS = "1"; $env:CUDA_VISIBLE_DEVICES = "-1"
$env:EXPERT_CACHE_PACKED = "E:\moe\35b-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"
$lim = Start-Process .venv\Scripts\python.exe -ArgumentList "-u", "scripts\ram_limit.py", "4.8", "2400" -PassThru -WindowStyle Hidden -RedirectStandardOutput logs\cache-lim.txt
$dog = Start-Job -ScriptBlock { param($limId, $games)
  while (Get-Process -Id $limId -ErrorAction SilentlyContinue) {
    if (Get-Process | Where-Object { $_.ProcessName -match $games }) { Stop-Process -Id $limId -Force; Get-Process llama-cli -ErrorAction SilentlyContinue | Stop-Process -Force; break }
    Start-Sleep 5 } } -ArgumentList $lim.Id, $games
Start-Sleep 60; Get-Content logs\cache-lim.txt -Tail 1
function Run($name, $cache, $buffered, $fast) {
  if (-not (Get-Process -Id $lim.Id -ErrorAction SilentlyContinue)) { "stopped early (a game started)"; return }
  $env:MOE_CACHE_GB = $cache
  if ($buffered) { $env:EXPERT_CACHE_BUFFERED = "1" } else { Remove-Item Env:EXPERT_CACHE_BUFFERED -ErrorAction SilentlyContinue }
  if ($fast) { $env:MOE_CACHE_BONUS = "1.0"; $env:MOE_BATCH_BONUS = "1.0" } else { Remove-Item Env:MOE_CACHE_BONUS, Env:MOE_BATCH_BONUS -ErrorAction SilentlyContinue }
  "=== $name  $(Get-Date -Format T)"
  $mem = Start-Job -ScriptBlock { $lo = 1e9; while ($true) { $a = (Get-Counter "\Memory\Available MBytes").CounterSamples[0].CookedValue; if ($a -lt $lo) { $lo = $a; Set-Content $env:TEMP\lowmem.txt $lo }; Start-Sleep 1 } }
  $p = Start-Process build-dist\bin\llama-cli.exe -ArgumentList $common -NoNewWindow -Wait -PassThru -RedirectStandardOutput "logs\cache-$name.out" -RedirectStandardError "logs\cache-$name.err"
  Stop-Job $mem; Remove-Job $mem -Force
  "  exit $($p.ExitCode), lowest available memory $(Get-Content $env:TEMP\lowmem.txt) MB"; Remove-Item $env:TEMP\lowmem.txt -ErrorAction SilentlyContinue
  Get-Content "logs\cache-$name.out", "logs\cache-$name.err" | Select-String "Generation|generation only: read|routing \(" | ForEach-Object { "  " + $_.Line.Trim().Substring(0, [Math]::Min(130, $_.Line.Trim().Length)) }
}
try {
  Run "c0.6" "0.6" $false $false
  Run "c1.2" "1.2" $false $false
  Run "c1.8" "1.8" $false $false
  Run "c2.4" "2.4" $false $false
  Run "c0.6-buffered" "0.6" $true $false
  Run "c1.2-buffered" "1.2" $true $false
  Run "c1.8-fast" "1.8" $false $true
} finally {
  Stop-Process $lim.Id -Force -ErrorAction SilentlyContinue
  Stop-Job $dog -ErrorAction SilentlyContinue; Remove-Job $dog -Force -ErrorAction SilentlyContinue
}
"EXIT done"
