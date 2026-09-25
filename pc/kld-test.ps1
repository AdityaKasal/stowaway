# Quality cost of using fewer experts per token on the 122B: KL divergence vs the normal 8 experts (WikiText-2).
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
New-Item -ItemType Directory -Force results\kld | Out-Null
$env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:MOE_CACHE_GB = "18"; $env:MOE_IO_THREADS = "6"; $env:MOE_PREGATE = "6"
$env:EXPERT_CACHE_PACKED = "models\122b\experts-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"
$env:EXPERT_CACHE_STRIPE = "E:\moe\experts-stripe"; $env:EXPERT_CACHE_TAIL = "E:\moe\experts-tail"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$base = @("-m", $m, "-ngl", "99", "--n-cpu-moe", "44", "--no-op-offload", "-t", "10", "-tb", "20", "-c", "512", "-b", "512",
          "--chunks", "8", "-f", "data\wikitext\wikitext-2-raw\wiki.test.raw")
function Run($name, $extra) {
  "`n=== $name  $(Get-Date -Format T)"
  $p = Start-Process build\bin\llama-perplexity.exe -ArgumentList ($base + $extra) -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "results\kld\$name.out.txt" -RedirectStandardError "results\kld\$name.err.txt"
  "exit $($p.ExitCode)"
  Get-Content "results\kld\$name.out.txt", "results\kld\$name.err.txt" |
    Select-String "Final estimate|Mean PPL|Mean    KLD|Same top p|Maximum KLD|99.0%   KLD|Median  KLD|RMS error" | ForEach-Object { $_.Line.Trim() }
}
Run "k8-base" @("--kl-divergence-base", "results\kld\base.kld")
foreach ($k in 7, 6, 5) {
  Run "k$k" @("--override-kv", "qwen35moe.expert_used_count=int:$k", "--kl-divergence-base", "results\kld\base.kld", "--kl-divergence")
}
"done $(Get-Date -Format T)"
