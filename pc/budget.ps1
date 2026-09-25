# Simulated budget laptop: no GPU, 4 threads, small memory budget, experts from the PCIe 3 drive (E:).
# usage: budget.ps1 <name> <cache_gb> <prompts.tsv> <n_predict> [extra flags...]
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$name, $gb, $prompts, $n = $args[0], $args[1], $args[2], $args[3]
$extra = @($args[4..($args.Count)] | Where-Object { $_ })
$env:CUDA_VISIBLE_DEVICES = "-1"           # no GPU at all
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$env:EXPERT_CACHE_PACKED = $(if ($env:BUDGET_PACKED) { $env:BUDGET_PACKED } else { "E:\moe\35b-packed" })
$env:EXPERT_CACHE_CHUNK_KB = "8192"
Remove-Item env:EXPERT_CACHE_STRIPE, env:EXPERT_CACHE_TAIL -ErrorAction SilentlyContinue
$m = $(if ($env:BUDGET_MODEL) { $env:BUDGET_MODEL } else { "models\Qwen3.5-35B-A3B-Q5_K_M.gguf" })
$flags = @("-ngl","0","--no-repack","--no-op-offload","-t","4","-tb","4","--expert-cache-gb",$gb,"--io-threads","4","--pregate","6","--no-hook","-n",$n) + $extra
# watch the process's real memory while it runs
$mem = Start-Job {
  [long]$peakWS = 0; [long]$peakPriv = 0; $seen = $false
  for ($i = 0; $i -lt 1200; $i++) {
    $p = Get-Process expert-logger -ErrorAction SilentlyContinue
    if ($p) { $seen = $true; $peakWS = [math]::Max($peakWS, [long]$p.WorkingSet64); $peakPriv = [math]::Max($peakPriv, [long]$p.PrivateMemorySize64) }
    elseif ($seen) { break }
    Start-Sleep -Milliseconds 500
  }
  "{0:N2} GB peak working set (all RAM the process touched), {1:N2} GB peak private" -f ($peakWS/1GB), ($peakPriv/1GB)
}
"=== $name  (cache $gb GB, extra: $extra)  $(Get-Date -Format T)"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m $prompts @flags | Select-String "exit"
Receive-Job $mem -Wait
$pr = Import-Csv "results\$name\prompts.csv"
"prompt reading: " + (($pr | ForEach-Object { "{0} tok {1} s" -f $_.n_prompt_tokens, $_.prefill_s }) -join ", ")
$t = Import-Csv "results\$name\tokens.csv" | Where-Object gen -eq 1 | ForEach-Object { [double]$_.ms } | Sort-Object
if ($t.Count -gt 10) { "median {0:N0} ms per token -> {1:N1} tok/s" -f $t[[int]($t.Count/2)], (1000/$t[[int]($t.Count/2)]) }
Get-Content "results\$name\log.txt" | Select-String "generation only|CUDA|no usable GPU|ggml_cuda_init" | Select-Object -First 3
