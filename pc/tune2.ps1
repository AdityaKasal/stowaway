# Round 2: whole-expert reads, and pre-gating (predict the next layer's experts and load them a layer early).
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$common = "-ngl 99 --n-cpu-moe 44 --no-op-offload --expert-cache-gb 19 -n 64 -t 10"
$configs = [ordered]@{
  "tune2-4mb"            = @("4096", "--no-hook")
  "tune2-1mb-pregate12"  = @("1024", "--no-log --pregate 12")
  "tune2-4mb-pregate12"  = @("4096", "--no-log --pregate 12")
  "tune2-4mb-pregate8"   = @("4096", "--no-log --pregate 8")
}
foreach ($name in $configs.Keys) {
  $env:EXPERT_CACHE_CHUNK_KB = $configs[$name][0]
  $flags = "$common $($configs[$name][1])"
  "`n=== $name  (chunk $($env:EXPERT_CACHE_CHUNK_KB) KB, $flags)  $(Get-Date -Format T)"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m bench-prompts.tsv @($flags -split " ") 2>&1 |
    Select-String "exit|SSD|^\["
  Get-Content "results\$name\log.txt" | Select-String "expert-cache: \d+ lookups|pre-gating: [0-9.]+%|error"
}
"tune2 done $(Get-Date -Format T)"
