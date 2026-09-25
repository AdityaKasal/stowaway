# 122B-A10B Q5_K_M (91.5 GB) with our expert cache vs. letting Windows page the file in.
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
# wait until nobody is gaming, or 21:00 at the latest
while ((Get-Process Spider-Man2 -ErrorAction SilentlyContinue) -and (Get-Date).Hour -lt 21) { Start-Sleep 60 }
$gaming = [bool](Get-Process Spider-Man2 -ErrorAction SilentlyContinue)
if ($gaming) { (Get-Process -Id $PID).PriorityClass = "BelowNormal"; "game still running: low priority" }
"start $(Get-Date -Format T)"

$env:LLAMA_NO_MMAP_PREFETCH = "1"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"

# how much RAM can the cache take? Windows refuses allocations beyond RAM + pagefile, so check both
$os = Get-CimInstance Win32_OperatingSystem
$freeGB   = $os.FreePhysicalMemory / 1MB
$commitGB = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB
$limitGB  = $os.TotalVirtualMemorySize / 1MB
$cache = [math]::Floor([math]::Min($freeGB - 5, $limitGB - $commitGB - 6))
$cache = [math]::Max([math]::Min($cache, 22), 4)
"free RAM {0:N1} GB, commit {1:N1}/{2:N1} GB -> expert cache {3} GB" -f $freeGB, $commitGB, $limitGB, $cache

function Run($name, $flags) {
  "`n=== $name  ($flags)  $(Get-Date -Format T)"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m bench-prompts.tsv @($flags -split " ") -n 96 2>&1 |
    Select-String "exit|SSD|^\[|expert-cache|error|memory"
}
$common = "--no-hook -ngl 99 --n-cpu-moe 44 --no-op-offload"
Run "122b-cache$cache"     "$common --expert-cache-gb $cache"
Run "122b-cache$([math]::Floor($cache/2))" "$common --expert-cache-gb $([math]::Floor($cache/2))"
Run "122b-os-paging"       "$common"

# router study on the 122B, with the cache so it finishes in reasonable time
"`n=== router logging, 24 prompts  $(Get-Date -Format T)"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 122b-router $m prompts.tsv -ngl 99 --n-cpu-moe 44 --no-op-offload --expert-cache-gb $cache -n 128 2>&1 |
  Select-String "exit|SSD|^\[|expert-cache|error"
.\.venv\Scripts\python analyze.py results\122b-router $m
"all done $(Get-Date -Format T)"
