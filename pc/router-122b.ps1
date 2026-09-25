# Router study on the 122B (24 prompts x 128 tokens), with the best cache settings from tuning.
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:LLAMA_NO_MMAP_PREFETCH = "1"
$env:EXPERT_CACHE_CHUNK_KB  = "4096"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
"start $(Get-Date -Format T)"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 122b-router $m prompts.tsv -ngl 99 --n-cpu-moe 44 --no-op-offload --expert-cache-gb 19 -t 10 -n 128 2>&1 |
  Select-String "exit|SSD|^\["
Get-Content results\122b-router\log.txt | Select-String "expert-cache: \d+ lookups|error"
.\.venv\Scripts\python analyze.py results\122b-router $m
"done $(Get-Date -Format T)"
