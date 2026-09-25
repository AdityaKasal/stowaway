# Final back-to-back comparison on the 122B (4 prompts x 128 tokens).
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$base = @("-ngl","99","--n-cpu-moe","44","--no-op-offload","-n","128","-t","10","--no-hook")
$runs = [ordered]@{
  "final-os-paging"   = @{ env = @{}; flags = @() }
  "final-cache-v1"    = @{ env = @{ EXPERT_CACHE_CHUNK_KB = "4096" }; flags = @("--expert-cache-gb","19") }
  "final-best-18g"    = @{ env = @{ EXPERT_CACHE_CHUNK_KB = "8192"; EXPERT_CACHE_PACKED = "models\122b\experts-packed" }; flags = @("--expert-cache-gb","18","--io-threads","4","--pregate","6") }
  "final-max-20g"     = @{ env = @{ EXPERT_CACHE_CHUNK_KB = "8192"; EXPERT_CACHE_PACKED = "models\122b\experts-packed" }; flags = @("--expert-cache-gb","20","--io-threads","4","--pregate","6") }
}
foreach ($name in $runs.Keys) {
  foreach ($k in "EXPERT_CACHE_CHUNK_KB","EXPERT_CACHE_PACKED") { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
  foreach ($k in $runs[$name].env.Keys) { Set-Item "env:$k" $runs[$name].env[$k] }
  "`n=== $name  $(Get-Date -Format T)"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m bench-prompts.tsv @($base + $runs[$name].flags) 2>&1 | Select-String "exit|^\["
  $t = Import-Csv "results\$name\tokens.csv" | Where-Object gen -eq 1 | ForEach-Object { [double]$_.ms } | Sort-Object
  "median {0:N0} ms per token -> {1:N1} tok/s" -f $t[[int]($t.Count / 2)], (1000 / $t[[int]($t.Count / 2)])
}
"`n--- identical output?"
$ref = Import-Csv results\final-os-paging\tokens.csv | Where-Object gen -eq 1 | ForEach-Object token_id
foreach ($name in @($runs.Keys)[1..3]) {
  $b = Import-Csv "results\$name\tokens.csv" | Where-Object gen -eq 1 | ForEach-Object token_id
  $same = 0; for ($i = 0; $i -lt [math]::Min($ref.Count, $b.Count); $i++) { if ($ref[$i] -eq $b[$i]) { $same++ } }
  "{0}: {1} of {2} tokens match the no-cache run" -f $name, $same, $ref.Count
}
"final done $(Get-Date -Format T)"
