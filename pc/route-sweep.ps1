# Cache-aware routing sweep: 35B Q5, one token at a time, 2.7 GB cache (8 GB laptop), KLD vs the unmodified model.
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }
$env:PATH = "C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
$root = "C:\Users\FSociety\moe-router-study"; Set-Location $root
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
cmake --build build-dist --target llama-perplexity -j 12 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 10
$out = "$root\results\route"; New-Item -ItemType Directory -Force $out | Out-Null
$env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:MOE_CACHE_GB = "2.7"; $env:MOE_IO_THREADS = "4"; $env:MOE_PREGATE = "6"; $env:MOE_STATS = "1"
$env:EXPERT_CACHE_PACKED = "E:\moe\35b-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"
$base = @("-m", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "-ngl", "0", "--no-repack", "--no-op-offload", "-t", "12", "-c", "512", "-b", "512",
          "-ub", "1", "--chunks", "4", "-f", "data\wikitext\wikitext-2-raw\wiki.test.raw", "--no-warmup")
function Run($name, $bonus, $extra) {
  "`n=== $name  $(Get-Date -Format T)"
  $env:MOE_CACHE_BONUS = $bonus
  $p = Start-Process build-dist\bin\llama-perplexity.exe -ArgumentList ($base + $extra) -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$out\$name.out.txt" -RedirectStandardError "$out\$name.err.txt"
  "exit $($p.ExitCode)"
  Get-Content "$out\$name.out.txt", "$out\$name.err.txt" | Select-String "Final estimate|Mean    KLD|Same top p:|99.0%   KLD|switched|generation only" | ForEach-Object { "  " + $_.Line.Trim() }
}
Run "base" "0" @("--kl-divergence-base", "$out\base.kld")
foreach ($b in "0.1", "0.25", "0.5", "1.0", "2.0") { Run "bonus-$b" $b @("--kl-divergence-base", "$out\base.kld", "--kl-divergence") }
"done $(Get-Date -Format T)"
