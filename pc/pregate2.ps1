# Hook-free pre-gating (predict from the MoE input inside the cache; no GPU syncs) vs the plain cache, 122B.
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$common = "-ngl 99 --n-cpu-moe 44 --no-op-offload --expert-cache-gb 19 -n 64 -t 10 --no-hook"
$configs = [ordered]@{
  "pg2-base-a"  = ""
  "pg2-m8-d1"   = "--pregate 8"
  "pg2-m6-d1"   = "--pregate 6"
  "pg2-m8-d2"   = "--pregate 8 --pregate-depth 2"
  "pg2-base-b"  = ""
}
foreach ($name in $configs.Keys) {
  $flags = "$common $($configs[$name])".Trim()
  "`n=== $name  ($($configs[$name]))  $(Get-Date -Format T)"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m bench-prompts.tsv @($flags -split " ") 2>&1 |
    Select-String "exit|^\["
  Get-Content "results\$name\log.txt" | Select-String "expert-cache: \d+ lookups|guesses:|still loading|error"
  $t = Import-Csv "results\$name\tokens.csv" | Where-Object gen -eq 1 | ForEach-Object { [double]$_.ms } | Sort-Object
  "median {0:N0} ms per token" -f $t[[int]($t.Count / 2)]
}
"pregate2 done $(Get-Date -Format T)"
