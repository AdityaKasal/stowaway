# Runs expert-logger with the given flags; results in results\<name>\ (prompts.csv has tok/s per prompt).
# usage: run-logger.ps1 <name> <model> <prompts.tsv> <extra flags...>
# (no param block: PowerShell would read "-n 64" as an abbreviation of -Name)
$Name, $Model, $Prompts, $rest = $args[0], $args[1], $args[2], $args[3..($args.Count - 1)]
$root = "C:\Users\FSociety\moe-router-study"
$out = "$root\results\$Name"
New-Item -ItemType Directory -Force $out | Out-Null
Remove-Item "$out\disk.csv" -ErrorAction SilentlyContinue
$perf = Start-Process typeperf -ArgumentList '"\PhysicalDisk(_Total)\Disk Read Bytes/sec" -si 1 -f CSV -o', "`"$out\disk.csv`"" -WindowStyle Hidden -PassThru
$t0 = Get-Date
# Start-Process keeps stderr as raw text (PowerShell 5's "2>" wraps it at console width)
$argv = @("-m", $Model, "--prompts", $Prompts, "--out-dir", $out, "-c", "4096", "-b", "512", "-fa", "on", "--seed", "1") + @($rest)
$p = Start-Process "$root\build\bin\expert-logger.exe" -ArgumentList $argv -NoNewWindow -Wait -PassThru `
  -RedirectStandardError "$out\log.txt" -RedirectStandardOutput "$out\stdout.txt"
"exit $($p.ExitCode) after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds
Stop-Process $perf -ErrorAction SilentlyContinue
Start-Sleep 1
$mbs = Import-Csv "$out\disk.csv" -ErrorAction SilentlyContinue | ForEach-Object { $v = ($_.PSObject.Properties | Select-Object -Last 1).Value; if ($v -match "^[0-9.]+$") { [double]$v / 1MB } } | Where-Object { $_ -gt 5 }
if ($mbs) { "SSD reads while busy: {0:N0} MB/s average over {1} s, {2:N1} GB total" -f ($mbs | Measure-Object -Average).Average, $mbs.Count, (($mbs | Measure-Object -Sum).Sum / 1024) }
Get-Content "$out\log.txt" | Select-String -Pattern "^\[|prefetch:|error|failed|out of memory"
