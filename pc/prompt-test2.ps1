$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$env:EXPERT_CACHE_PACKED = "models\122b\experts-packed"
$env:EXPERT_CACHE_CHUNK_KB = "8192"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$flags = @("--no-hook","-ngl","99","--n-cpu-moe","44","--no-op-offload","--expert-cache-gb","18","--io-threads","4","--pregate","6","-t","10","-tb","20")
foreach ($cfg in @(@("pp2-long-all", "0", "long-prompt.tsv", "8"), @("pp2-long-s64", "64", "long-prompt.tsv", "8"), @("pp2-long-s32", "32", "long-prompt.tsv", "8"), @("pp2-short-all", "0", "bench-prompts.tsv", "16"))) {
  $name, $tok, $p, $n = $cfg
  $env:PREGATE_PROMPT_TOKENS = $tok
  "`n=== $name (sampled tokens: $tok)  $(Get-Date -Format T)"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m $p @($flags + @("-n", $n)) | Select-String "exit"
  $pr = Import-Csv "results\$name\prompts.csv"
  "prompt reading: " + (($pr | ForEach-Object { "{0} tok in {1} s" -f $_.n_prompt_tokens, $_.prefill_s }) -join ", ") + ("  | total {0:N1} s" -f ($pr | Measure-Object prefill_s -Sum).Sum)
  Get-Content "results\$name\log.txt" | Select-String "expert-cache: \d+ lookups"
}
foreach ($pair in @(@("pp-long-off","pp2-long-all"), @("pp-long-off","pp2-long-s32"), @("pp-short-off","pp2-short-all"))) {
  $a = Import-Csv "results\$($pair[0])\tokens.csv" | Where-Object gen -eq 1 | ForEach-Object token_id
  $b = Import-Csv "results\$($pair[1])\tokens.csv" | Where-Object gen -eq 1 | ForEach-Object token_id
  $same = 0; for ($i = 0; $i -lt [math]::Min($a.Count, $b.Count); $i++) { if ($a[$i] -eq $b[$i]) { $same++ } }
  "{0} vs {1}: {2} of {3} tokens match" -f $pair[0], $pair[1], $same, $a.Count
}
