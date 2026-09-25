# 122B-A10B Q5_K_M (91.5 GB) on a 32 GB RAM + 16 GB VRAM PC: the model does not fit, experts stream from SSD.
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
while ((Get-ScheduledTask moe-download-big).State -eq "Running") { Start-Sleep 30 }
"download finished $(Get-Date -Format T)"
Get-Content logs\download-big.log -Tail 4
.\.venv\Scripts\python model_info.py models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf

$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
function Run($name, $flags) {
  "`n=== $name  ($flags)  $(Get-Date -Format T)"
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-logger.ps1 $name $m bench-prompts.tsv @($flags -split " ") -n 96 2>&1 | Select-String "exit|SSD|^\[|prefetch|error|memory"
}
# find how many layers of experts fit on the GPU (always-needed weights go there first)
foreach ($n in 44, 42, 40) {
  Run "122b-fit-ncmoe$n" "--no-hook -ngl 99 --n-cpu-moe $n"
}
"fit test done $(Get-Date -Format T)"
