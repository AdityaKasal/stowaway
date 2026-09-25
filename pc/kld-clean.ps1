# Clean re-measure after the order-preserving fix: Qwen 35B (Windows) and gpt-oss-20b (Linux VM)
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }
$env:PATH = "C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
$root = "C:\Users\FSociety\moe-router-study"; Set-Location $root
cmake --build build-dist --target llama-perplexity llama-cli llama-server -j 12 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 5
wsl -d Ubuntu-22.04 -u root -e bash -c "cd /root/moe && export PATH=/usr/local/bin:`$PATH && rsync -a /mnt/c/Users/FSociety/moe-router-study/llama.cpp/common/moe-stream.h llama.cpp/common/ && cmake --build build-dist --target llama-perplexity llama-cli llama-server -j 16 2>&1 | tail -1 && cp build-dist/bin/llama-perplexity build-dist/bin/llama-cli build-dist/bin/llama-server /mnt/c/Users/FSociety/moe-router-study/dist/linux-test/" 2>&1 | Select-Object -Last 1
"##### gpt-oss-20b"
wsl -d Ubuntu -u root -e bash -c "cp /mnt/c/Users/FSociety/moe-router-study/dist/linux-test/* /root/moe/app/ && bash /mnt/c/Users/FSociety/moe-router-study/vm/gptoss-kld3.sh"
"##### Qwen 35B"
$out = "$root\results\route"
$env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:MOE_CACHE_GB = "2.7"; $env:MOE_IO_THREADS = "4"; $env:MOE_PREGATE = "6"; $env:MOE_STATS = "1"
$env:EXPERT_CACHE_PACKED = "E:\moe\35b-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"; $env:CUDA_VISIBLE_DEVICES = "-1"
$base = @("-m", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "-ngl", "0", "--no-repack", "--no-op-offload", "-t", "12", "-c", "512", "-b", "512",
          "-ub", "1", "--chunks", "4", "-f", "data\wikitext\wikitext-2-raw\wiki.test.raw", "--no-warmup", "--kl-divergence-base", "$out\base.kld", "--kl-divergence")
foreach ($c in @(@("clean-control", "1.0", "8"), @("clean-bonus-1.0", "1.0", "0"), @("clean-bonus-0.5", "0.5", "0"))) {
  $env:MOE_CACHE_BONUS = $c[1]; $env:MOE_CACHE_PROTECT = $c[2]
  "=== $($c[0])  $(Get-Date -Format T)"
  $p = Start-Process build-dist\bin\llama-perplexity.exe -ArgumentList $base -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$out\$($c[0]).out.txt" -RedirectStandardError "$out\$($c[0]).err.txt"
  Get-Content "$out\$($c[0]).out.txt", "$out\$($c[0]).err.txt" | Select-String "Mean    KLD|Same top p:|switched|generation only: read" | ForEach-Object { "  " + $_.Line.Trim().Substring(0, [Math]::Min(140, $_.Line.Trim().Length)) }
}
"EXIT done"
