# Timing breakdown for the 122B cache. usage: diag.ps1 <name> [chunk_kb] [extra flags...]
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$name = $args[0]; $chunk = $args[1]; $extra = @($args[2..($args.Count)] | Where-Object { $_ })
# optional env overrides passed as NAME=VALUE in the extra args
$extra = @($extra | ForEach-Object { if ($_ -match "^([A-Z_]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2]; "SETENV" } else { $_ } } | Where-Object { $_ -ne "SETENV" })
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$env:EXPERT_CACHE_CHUNK_KB = $chunk
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$flags = @("-ngl","99","--n-cpu-moe","44","--no-op-offload","--expert-cache-gb","19","-n","64","-t","10","--no-hook") + $extra
"=== $name (chunk $chunk KB, extra: $extra) $(Get-Date -Format T)"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m bench-prompts.tsv @flags 2>&1 | Select-String "exit|^\["
Get-Content "results\$name\log.txt" | Select-String "expert-cache:" | Where-Object { $_ -notmatch "GB for" }
$t = Import-Csv "results\$name\tokens.csv" | Where-Object gen -eq 1 | ForEach-Object { [double]$_.ms } | Sort-Object
"median {0:N0} ms per token" -f $t[[int]($t.Count / 2)]
