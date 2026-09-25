# Does cache-aware routing (bonus 1.0) cut reads when the cache is small? 35B Q5, one token at a time, 1 x 512 tokens.
$root = "C:\Users\FSociety\moe-router-study"; Set-Location $root
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
$out = "$root\results\route-threshold"; New-Item -ItemType Directory -Force $out | Out-Null
$env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:MOE_IO_THREADS = "4"; $env:MOE_PREGATE = "6"; $env:MOE_STATS = "1"
$env:EXPERT_CACHE_PACKED = "E:\moe\35b-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"; $env:CUDA_VISIBLE_DEVICES = "-1"
$base = @("-m", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "-ngl", "0", "--no-repack", "--no-op-offload", "-t", "12",
          "-c", "512", "-b", "512", "-ub", "1", "--chunks", "1", "-f", "data\wikitext\wikitext-2-raw\wiki.test.raw", "--no-warmup")
foreach ($gb in "0.37", "0.75", "1.5") { foreach ($b in "0", "1.0") {
  $env:MOE_CACHE_GB = $gb; $env:MOE_CACHE_BONUS = $b
  $p = Start-Process build-dist\bin\llama-perplexity.exe -ArgumentList $base -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$out\c$gb-b$b.out.txt" -RedirectStandardError "$out\c$gb-b$b.err.txt"
  $g = (Select-String -Path "$out\c$gb-b$b.err.txt" -Pattern "generation only: read ([0-9.]+) GB").Matches.Groups[1].Value
  $sw = (Select-String -Path "$out\c$gb-b$b.err.txt" -Pattern "\(([0-9.]+)%\)$" | Select-Object -Last 1).Matches.Groups[1].Value
  "cache $gb GB, bonus $b : generation read $g GB, picks switched $sw%"
} }
"EXIT done"
