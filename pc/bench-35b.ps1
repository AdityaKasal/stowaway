# 35B-A3B Q5 on the PC: how many layers' experts should stay in RAM (-ncmoe) vs go on the 16 GB GPU?
$root = "C:\Users\FSociety\moe-router-study"
while ((Get-ScheduledTask moe-download).State -eq "Running") { Start-Sleep 30 }
$m = "$root\models\Qwen3.5-35B-A3B-Q5_K_M.gguf"
"model: " + (Get-Item $m).Length + " bytes  $(Get-Date -Format T)"
$b = "$root\build\bin\llama-bench.exe"

"`n## GPU + RAM split (ncmoe = layers whose experts stay in RAM; lower = more on GPU)"
& $b -m $m -ngl 99 -ncmoe 16,18,20,22,24,28,40 -fa on -p 512 -n 128 -r 2 -o md 2>&1 | Select-String "^\|"
"`n## CPU only (no GPU), for comparison with the Mac"
& $b -m $m -ngl 0 -fa on -p 512 -n 128 -r 2 -o md 2>&1 | Select-String "^\|"
"`n## everything on GPU (does not fit: expected to fail)"
& $b -m $m -ngl 99 -ncmoe 0 -fa on -p 512 -n 128 -r 1 -o md 2>&1 | Select-Object -Last 6
"done $(Get-Date -Format T)"
