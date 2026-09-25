# 122B on the "8 GB laptop": speculative decoding with the 0.8B helper vs none. Greedy, so outputs must match.
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:CUDA_VISIBLE_DEVICES = "-1"; $env:LLAMA_NO_MMAP_PREFETCH = "1"
$env:EXPERT_CACHE_PACKED = "models\122b\experts-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"
$env:MOE_CACHE_GB = "0.5"; $env:MOE_IO_THREADS = "4"; $env:MOE_PREGATE = "0"
$m = "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf"
$prompt = $env:SPEC_PROMPT
$base = @("-m", $m, "-ngl", "0", "--no-repack", "--no-op-offload", "-c", "4096", "-b", "128", "-ub", "128", "-t", "4", "-tb", "4",
          "-n", "64", "--temp", "0", "--seed", "1", "-p", ("`"" + $prompt + "`""), "-st", "--simple-io", "--no-display-prompt")
function Run($name, $dense, $spec, $packed) {
  $env:EXPERT_CACHE_DENSE_GB = $dense
  if ($packed) { $env:EXPERT_CACHE_DENSE_PACKED = "models\122b\dense-packed" } else { Remove-Item env:EXPERT_CACHE_DENSE_PACKED -ErrorAction SilentlyContinue }
  $mem = Start-Job { [long]$pk = 0; $seen = $false; for ($i = 0; $i -lt 2400; $i++) { $p = Get-Process llama-cli -ErrorAction SilentlyContinue; if ($p) { $seen = $true; $pk = [math]::Max($pk, [long]$p.PrivateMemorySize64) } elseif ($seen) { break }; Start-Sleep -Milliseconds 500 }; "{0:N2} GB peak private" -f ($pk / 1GB) }
  "`n=== $name (locked $dense GB, $spec)  $(Get-Date -Format T)"
  $p = Start-Process build\bin\llama-cli.exe -ArgumentList ($base + $spec) -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "results\$name.out.txt" -RedirectStandardError "results\$name.err.txt"
  Receive-Job $mem -Wait
  Get-Content "results\$name.out.txt" | Select-String "Generation:|Prompt:"
  Get-Content "results\$name.err.txt" | Select-String "accept|draft.*n_|statistics" | Select-Object -First 4
}
$draft = @("-md", "models\Qwen3.5-0.8B-Q4_K_M.gguf", "-ngld", "0", "--spec-type", "draft-simple")
Run "$($env:SPEC_TAG)-none"  "3.2" @("--no-warmup") $true
Run "$($env:SPEC_TAG)-best"  "2.6" ($draft + @("--spec-draft-n-max", "12", "--spec-draft-p-min", "0.8", "--no-warmup")) $true
$a = (Get-Content "results\$($env:SPEC_TAG)-none.out.txt" -Raw) -replace "\[ Prompt.*", ""
$b = (Get-Content "results\$($env:SPEC_TAG)-best.out.txt" -Raw) -replace "\[ Prompt.*", ""
"same text: " + ($a -eq $b)
