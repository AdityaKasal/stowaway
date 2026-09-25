# Does the improved pre-gating beat the plain cache on the 122B? Baseline runs first and last to catch drift.
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$common = "-ngl 99 --n-cpu-moe 44 --no-op-offload --expert-cache-gb 19 -n 64 -t 10"
$configs = [ordered]@{
  "pg-base-a"   = "--no-hook"
  "pg-m8-d1"    = "--no-log --pregate 8"
  "pg-m12-d1"   = "--no-log --pregate 12"
  "pg-m8-d2"    = "--no-log --pregate 8 --pregate-depth 2"
  "pg-m6-d2"    = "--no-log --pregate 6 --pregate-depth 2"
  "pg-base-b"   = "--no-hook"
}
foreach ($name in $configs.Keys) {
  $flags = "$common $($configs[$name])"
  "`n=== $name  ($($configs[$name]))  $(Get-Date -Format T)"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m bench-prompts.tsv @($flags -split " ") 2>&1 |
    Select-String "exit|^\["
  Get-Content "results\$name\log.txt" | Select-String "expert-cache: \d+ lookups|guesses:|pre-gating: [0-9.]+%|error"
}
"pregate done $(Get-Date -Format T)"
