# MTP versions (built-in multi-token prediction head): 35B first to validate, then the 122B.
New-Item -ItemType Directory -Force D:\moe\122b-mtp | Out-Null
function Get-File($url, $dst) {
  for ($try = 0; $try -lt 30; $try++) {
    curl.exe -L --fail -sS --retry 5 -C - -o $dst $url
    if ($LASTEXITCODE -eq 0) { break }
    "curl exit $LASTEXITCODE on $dst, resuming"; Start-Sleep 15
  }
  "{0} done: {1:N1} GB  {2}" -f (Split-Path $dst -Leaf), ((Get-Item $dst).Length / 1e9), (Get-Date -Format T)
}
Get-File "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q5_K_M.gguf" "D:\moe\Qwen3.5-35B-A3B-MTP-UD-Q5_K_M.gguf"
"35B ready"
foreach ($i in 1..3) {
  $f = "Qwen3.5-122B-A10B-UD-Q5_K_M-0000$i-of-00003.gguf"
  Get-File "https://huggingface.co/unsloth/Qwen3.5-122B-A10B-MTP-GGUF/resolve/main/UD-Q5_K_M/$f" "D:\moe\122b-mtp\$f"
}
"122B ready"
