# One drive vs two drives, 122B, same seed. 4 prompts x 128 tokens, then the long prompt.
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$env:EXPERT_CACHE_PACKED = "models\122b\experts-packed"
$env:EXPERT_CACHE_CHUNK_KB = "8192"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$base = @("--no-hook","-ngl","99","--n-cpu-moe","44","--no-op-offload","--expert-cache-gb","18","--pregate","6","-t","10","-tb","20")
function Run($name, $stripe, $io, $prompts, $n) {
  if ($stripe) { $env:EXPERT_CACHE_STRIPE = "E:\moe\experts-stripe" } else { Remove-Item env:EXPERT_CACHE_STRIPE -ErrorAction SilentlyContinue }
  "`n=== $name (two drives: $stripe, I/O threads: $io)  $(Get-Date -Format T)"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m $prompts @($base + @("--io-threads", $io, "-n", $n)) | Select-String "exit|^\["
  $pr = Import-Csv "results\$name\prompts.csv"
  "prompt reading total {0:N1} s" -f ($pr | Measure-Object prefill_s -Sum).Sum
  $t = Import-Csv "results\$name\tokens.csv" | Where-Object gen -eq 1 | ForEach-Object { [double]$_.ms } | Sort-Object
  if ($t.Count -gt 20) { "median {0:N0} ms per token -> {1:N1} tok/s" -f $t[[int]($t.Count / 2)], (1000 / $t[[int]($t.Count / 2)]) }
  Get-Content "results\$name\log.txt" | Select-String "second drive|generation only"
}
Run "st-one-io4"  $false "4" "bench-prompts.tsv" "128"
Run "st-two-io4"  $true  "4" "bench-prompts.tsv" "128"
Run "st-two-io6"  $true  "6" "bench-prompts.tsv" "128"
Run "st-two-io8"  $true  "8" "bench-prompts.tsv" "128"
Run "st-one-long" $false "4" "long-prompt.tsv" "8"
Run "st-two-long" $true  "6" "long-prompt.tsv" "8"
"`n--- same output?"
$ref = Import-Csv results\st-one-io4\tokens.csv | Where-Object gen -eq 1 | ForEach-Object token_id
foreach ($name in "st-two-io4","st-two-io6","st-two-io8") {
  $b = Import-Csv "results\$name\tokens.csv" | Where-Object gen -eq 1 | ForEach-Object token_id
  $same = 0; for ($i = 0; $i -lt [math]::Min($ref.Count, $b.Count); $i++) { if ($ref[$i] -eq $b[$i]) { $same++ } }
  "{0}: {1} of {2} tokens match the one-drive run" -f $name, $same, $ref.Count
}
