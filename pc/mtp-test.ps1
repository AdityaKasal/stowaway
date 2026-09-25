# Built-in MTP head vs the 0.8B helper vs nothing, on the 35B MTP model. Budget-like: no GPU, 4 threads, 3 GB/s reads.
$root = "C:\Users\FSociety\moe-router-study"
Set-Location $root
$env:CUDA_VISIBLE_DEVICES = "-1"; $env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:EXPERT_CACHE_MAX_MBPS = "3000"
$env:EXPERT_CACHE_PACKED = $env:MTP_PACKED; $env:EXPERT_CACHE_CHUNK_KB = "8192"
$env:MOE_CACHE_GB = $env:MTP_CACHE; $env:MOE_IO_THREADS = "4"; $env:MOE_PREGATE = "0"
if ($env:MTP_DENSE) { $env:EXPERT_CACHE_DENSE_GB = $env:MTP_DENSE } else { Remove-Item env:EXPERT_CACHE_DENSE_GB -ErrorAction SilentlyContinue }
$m = $env:MTP_MODEL
function Run($name, $prompt, $spec) {
  $args0 = @("-m", $m, "-ngl", "0", "--no-repack", "--no-op-offload", "-c", "4096", "-b", "128", "-ub", "128", "-t", "4", "-tb", "4",
             "-n", "96", "--temp", "0", "--seed", "1", "-p", ("`"" + $prompt + "`""), "-st", "--simple-io", "--no-display-prompt", "--no-warmup") + $spec
  $p = Start-Process build\bin\llama-cli.exe -ArgumentList $args0 -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "results\$name.out.txt" -RedirectStandardError "results\$name.err.txt"
  $g = Get-Content "results\$name.out.txt" | Select-String "Generation:" | ForEach-Object { $_.Line.Trim() }
  "{0,-22} exit {1}  {2}" -f $name, $p.ExitCode, $g
  if ($p.ExitCode -ne 0) { Get-Content "results\$name.err.txt" -Tail 4 }
}
$helper = @("-md", "models\Qwen3.5-0.8B-Q4_K_M.gguf", "-ngld", "0", "--spec-type", "draft-simple", "--spec-draft-n-max", "12", "--spec-draft-p-min", "0.8")
$prompts = @{ "story" = "Write the opening two sentences of a mystery story set in a lighthouse."; "code" = "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring." }
foreach ($pn in "story", "code") {
  $t = $env:MTP_TAG + "-" + $pn
  Run "$t-none"   $prompts[$pn] @()
  Run "$t-helper" $prompts[$pn] $helper
  foreach ($n in 2, 3, 4) { Run "$t-mtp$n" $prompts[$pn] @("--spec-type", "draft-mtp", "--spec-draft-n-max", "$n") }
  $ref = (Get-Content "results\$t-none.out.txt" -Raw) -replace "\[ Prompt.*", ""
  foreach ($v in "helper", "mtp2", "mtp3", "mtp4") { $o = (Get-Content "results\$t-$v.out.txt" -Raw -ErrorAction SilentlyContinue) -replace "\[ Prompt.*", ""; "  $v same text: " + ($o -eq $ref) }
}
