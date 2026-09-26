# one dense-on-GPU run of the 35B with GPU usage logged every second (logger runs inside this task so it stays alive)
$root = "C:\Users\FSociety\moe-router-study"; Set-Location $root
$csv = "$root\logs\gpu-usage.csv"; Remove-Item $csv -ErrorAction SilentlyContinue
$log = Start-Job -ScriptBlock { param($csv)  # one sample per second, written at once
  while ($true) { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | Add-Content $csv; Start-Sleep 1 } } -ArgumentList $csv
Start-Sleep 3
$story = "Write the opening two sentences of a mystery story set in a lighthouse."
$env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:MOE_IO_THREADS = "4"; $env:MOE_PREGATE = "6"; $env:MOE_CACHE_GB = "2.5"
$env:EXPERT_CACHE_PACKED = "E:\moe\35b-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"; $env:CUDA_VISIBLE_DEVICES = "0"
$args2 = @("-m", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "--no-repack", "--no-op-offload", "-c", "2048", "-b", "64", "-ub", "64",
           "-t", "4", "-tb", "4", "-n", "128", "--no-warmup", "-rea", "off", "-p", "`"$story`"", "-st", "--simple-io", "--no-display-prompt", "-ngl", "99", "--n-cpu-moe", "99")
$p = Start-Process build\bin\llama-cli.exe -ArgumentList $args2 -NoNewWindow -Wait -PassThru -RedirectStandardOutput logs\gpuw.out -RedirectStandardError logs\gpuw.err
Start-Sleep 3; Stop-Job $log; Remove-Job $log -Force
"run exit $($p.ExitCode)"; Get-Content logs\gpuw.out, logs\gpuw.err | Select-String "Generation|CUDA0 model buffer size|CUDA0 KV buffer" | Select-Object -First 3 | ForEach-Object { "  " + $_.Line.Trim() }
$rows = Get-Content $csv | ForEach-Object { $f = $_ -split ",\s*"; [pscustomobject]@{ util = [double]$f[1]; mem = [double]$f[2]; temp = [double]$f[3]; watts = [double]$f[4] } }
"samples: $($rows.Count) (1 per second)"
"GPU busy: average {0:N0}%, peak {1:N0}%" -f ($rows | Measure-Object util -Average).Average, ($rows | Measure-Object util -Maximum).Maximum
"GPU memory: peak {0:N0} MB of 16311" -f ($rows | Measure-Object mem -Maximum).Maximum
"temperature: peak {0:N0} C; power: average {1:N0} W, peak {2:N0} W" -f ($rows | Measure-Object temp -Maximum).Maximum, ($rows | Measure-Object watts -Average).Average, ($rows | Measure-Object watts -Maximum).Maximum
"EXIT done"
