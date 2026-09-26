# Would a small GPU help an 8 GB laptop? 35B Q5, ~4.8 GB free (ram_limit), experts streamed from E: either way.
# A: CPU only (today's plan). B: always-needed weights on the GPU (-ngl 99, all experts kept on the CPU), bigger cache.
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }
$cuda = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
$env:PATH = "$cuda\bin;C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
$root = "C:\Users\FSociety\moe-router-study"; Set-Location $root
"build (CUDA) $(Get-Date -Format T)"
cmake --build build --target llama-cli -j 8 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 5
$story = "Write the opening two sentences of a mystery story set in a lighthouse."
$common = @("-m", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "--no-repack", "--no-op-offload", "-c", "2048", "-b", "64", "-ub", "64",
            "-t", "4", "-tb", "4", "-n", "64", "--no-warmup", "-rea", "off", "-p", "`"$story`"", "-st", "--simple-io", "--no-display-prompt")  # Start-Process splits unquoted spaces
$env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:MOE_IO_THREADS = "4"; $env:MOE_PREGATE = "6"; $env:MOE_STATS = "1"
$env:EXPERT_CACHE_PACKED = "E:\moe\35b-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"
$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
$lim = Start-Process .venv\Scripts\python.exe -ArgumentList "-u", "scripts\ram_limit.py", "4.8", "1800" -PassThru -WindowStyle Hidden -RedirectStandardOutput logs\gpu-lim.txt
# watchdog: if a game starts while memory is locked, free it at once and stop the test
$dog = Start-Job -ScriptBlock { param($limId, $games)
  while (Get-Process -Id $limId -ErrorAction SilentlyContinue) {
    if (Get-Process | Where-Object { $_.ProcessName -match $games }) {
      Stop-Process -Id $limId -Force -ErrorAction SilentlyContinue
      Get-Process llama-cli -ErrorAction SilentlyContinue | Stop-Process -Force
      "game started: freed the memory and stopped"; break }
    Start-Sleep 5 } } -ArgumentList $lim.Id, $games
Start-Sleep 60; Get-Content logs\gpu-lim.txt -Tail 1
try {
function Run($name, $cache, $extra, $cudaVis) {
  $env:MOE_CACHE_GB = $cache; $env:CUDA_VISIBLE_DEVICES = $cudaVis
  "=== $name (cache $cache GB)  $(Get-Date -Format T)"
  $p = Start-Process build\bin\llama-cli.exe -ArgumentList ($common + $extra) -NoNewWindow -Wait -PassThru -RedirectStandardOutput "logs\gpu-$name.out" -RedirectStandardError "logs\gpu-$name.err"
  "  exit $($p.ExitCode)"
  Get-Content "logs\gpu-$name.out", "logs\gpu-$name.err" | Select-String "Generation|generation only: read|offloaded|CUDA0 model buffer|out of memory" | Select-Object -First 5 | ForEach-Object { "  " + $_.Line.Trim().Substring(0, [Math]::Min(140, $_.Line.Trim().Length)) }
}
foreach ($i in 1, 2) {
  if (-not (Get-Process -Id $lim.Id -ErrorAction SilentlyContinue)) { "stopped early (a game started)"; break }
  Run "cpu-$i" "0.6" @("-ngl", "0") "-1"
  Run "gpu-dense-$i" "2.5" @("-ngl", "99", "--n-cpu-moe", "99") "0"
}
} finally {
  Stop-Process $lim.Id -Force -ErrorAction SilentlyContinue   # always give the memory back
  Receive-Job $dog -ErrorAction SilentlyContinue; Stop-Job $dog -ErrorAction SilentlyContinue; Remove-Job $dog -Force -ErrorAction SilentlyContinue
}
"EXIT done"
