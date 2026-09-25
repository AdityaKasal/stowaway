# Tune the expert cache on the 122B: compute threads vs I/O threads vs read size. 4 prompts x 64 tokens each.
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$common = "--no-hook -ngl 99 --n-cpu-moe 44 --no-op-offload --expert-cache-gb 19 -n 64"
$configs = [ordered]@{
  "tune-default"      = @("1024", "")
  "tune-t10"          = @("1024", "-t 10")
  "tune-io32-256k"    = @("256",  "--io-threads 32")
  "tune-t10-io32-512k"= @("512",  "-t 10 --io-threads 32")
}
foreach ($name in $configs.Keys) {
  $env:EXPERT_CACHE_CHUNK_KB = $configs[$name][0]
  $flags = "$common $($configs[$name][1])".Trim()
  "`n=== $name  (chunk $($env:EXPERT_CACHE_CHUNK_KB) KB, $flags)  $(Get-Date -Format T)"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m bench-prompts.tsv @($flags -split " ") 2>&1 |
    Select-String "exit|SSD|^\["
  Get-Content "results\$name\log.txt" | Select-String "expert-cache: \d+ lookups|model waited"
}
"tune done $(Get-Date -Format T)"
