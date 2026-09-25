# Batch-aware routing for prompts: 122B Q5, question-sized batches (48 tokens), KLD vs no bonus, and distinct experts read.
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }
$env:PATH = "C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
$root = "C:\Users\FSociety\moe-router-study"; Set-Location $root
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
cmake --build build-dist --target llama-perplexity llama-cli llama-server -j 12 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 10
$out = "$root\results\batchroute"; New-Item -ItemType Directory -Force $out | Out-Null
$env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:MOE_CACHE_GB = "3"; $env:MOE_IO_THREADS = "6"; $env:MOE_PREGATE = "0"; $env:MOE_STATS = "1"
$env:EXPERT_CACHE_PACKED = "models\122b\experts-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"; $env:CUDA_VISIBLE_DEVICES = "-1"
$base = @("-m", "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf", "-ngl", "0", "--no-repack", "--no-op-offload", "-t", "12",
          "-c", "512", "-b", "512", "-ub", "48", "--chunks", "3", "-f", "data\wikitext\wikitext-2-raw\wiki.test.raw", "--no-warmup")
function Run($name, $bonus, $extra) {
  "`n=== $name  $(Get-Date -Format T)"
  $env:MOE_BATCH_BONUS = $bonus
  $t0 = Get-Date
  $p = Start-Process build-dist\bin\llama-perplexity.exe -ArgumentList ($base + $extra) -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$out\$name.out.txt" -RedirectStandardError "$out\$name.err.txt"
  "exit $($p.ExitCode), $([int]((Get-Date) - $t0).TotalSeconds) s"
  Get-Content "$out\$name.out.txt", "$out\$name.err.txt" | Select-String "Final estimate|Mean    KLD|Same top p:|batch-aware|read [0-9.]+ GB from disk" | ForEach-Object { "  " + $_.Line.Trim() }
}
Run "base" "0" @("--kl-divergence-base", "$out\base.kld")
foreach ($b in "0.25", "0.5", "1.0", "2.0") { Run "batch-$b" $b @("--kl-divergence-base", "$out\base.kld", "--kl-divergence") }
"EXIT done"
