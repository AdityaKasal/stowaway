$ErrorActionPreference = "Stop"
$root = "C:\Users\FSociety\moe-router-study"
New-Item -ItemType Directory -Force "$root\models", "$root\bin", "$root\zips" | Out-Null
$rel = "https://github.com/ggml-org/llama.cpp/releases/download/b11149"
foreach ($z in "llama-b11149-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip") {
  "downloading $z"
  curl.exe -L --fail -sS --retry 5 -C - -o "$root\zips\$z" "$rel/$z"
  if ($LASTEXITCODE) { throw "download failed: $z" }
  Expand-Archive -Force "$root\zips\$z" "$root\bin"
}
"llama.cpp ready"
$m = "Qwen3.5-35B-A3B-Q5_K_M.gguf"
for ($i = 0; $i -lt 20; $i++) {
  curl.exe -L --fail -sS --retry 5 -C - -o "$root\models\$m" "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/$m"
  if ($LASTEXITCODE -eq 0) { break }
  "curl exit $LASTEXITCODE, resuming"
  Start-Sleep 10
}
"model size: " + (Get-Item "$root\models\$m").Length
