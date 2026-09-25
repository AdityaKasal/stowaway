# 122B-A10B at Q5_K_M (91.5 GB, 3 parts). Waits for the 35B download so they don't split bandwidth.
$root = "C:\Users\FSociety\moe-router-study"
while ((Get-ScheduledTask moe-download).State -eq "Running") { Start-Sleep 30 }
"35B download finished, starting 122B $(Get-Date -Format T)"
New-Item -ItemType Directory -Force "$root\models\122b" | Out-Null
$base = "https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/main/Q5_K_M"
foreach ($i in 1..3) {
  $f = "Qwen3.5-122B-A10B-Q5_K_M-0000$i-of-00003.gguf"
  for ($try = 0; $try -lt 30; $try++) {
    curl.exe -L --fail -sS --retry 5 -C - -o "$root\models\122b\$f" "$base/$f"
    if ($LASTEXITCODE -eq 0) { break }
    "curl exit $LASTEXITCODE on $f, resuming"
    Start-Sleep 15
  }
  "$f done: " + (Get-Item "$root\models\122b\$f").Length + "  $(Get-Date -Format T)"
}
"all done"
