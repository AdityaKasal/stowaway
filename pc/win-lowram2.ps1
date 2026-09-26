# The released Windows app under memory pressure (ram_limit.py locks memory away): like 8 GB and ~4 GB Windows machines
$root = "C:\Users\FSociety\moe-router-study"; Set-Location $root
$t = "C:\Users\FSociety\stowaway-winlow"; Remove-Item -Recurse -Force $t -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force $t | Out-Null
Invoke-WebRequest -UseBasicParsing "https://github.com/AdityaKasal/stowaway/releases/latest/download/stowaway-windows.zip" -OutFile "$t\s.zip"
Expand-Archive "$t\s.zip" -DestinationPath $t
$app = "$t\stowaway-windows\stowaway.exe"; & $app --version
$env:STOWAWAY_NO_UPDATE_CHECK = "1"; $env:MOE_HOME = "$t\home"; $env:MOE_STATS = "1"
$story = "Write the opening two sentences of a mystery story set in a lighthouse."
function Run($name, $argl) {
  "=== $name  $(Get-Date -Format T)"
  & $app @argl *> "$t\$name.txt"
  "  exit $LASTEXITCODE"
  Select-String -Path "$t\$name.txt" -Pattern "^machine|^small|^plan|Generation|can't run|rror" | ForEach-Object { "  " + $_.Line.Substring(0, [Math]::Min(150, $_.Line.Length)) }
}
foreach ($leave in @("4.5")) {
  "##### about $leave GB free"
  $lim = Start-Process .venv\Scripts\python.exe -ArgumentList "-u", "scripts\ram_limit.py", $leave, "3600" -PassThru -WindowStyle Hidden -RedirectStandardOutput "$t\lim-$leave.txt"
  Start-Sleep 60
  Get-Content "$t\lim-$leave.txt" -Tail 1
  Run "q35-default-a" @("run", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "--packed", "E:\moe\35b-packed", "-n", "64", "-p", $story)
  Run "q35-smallmode" @("run", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "--packed", "E:\moe\35b-packed", "-n", "64", "--ram", "4.9", "-p", $story)
  Run "q35-default-b" @("run", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "--packed", "E:\moe\35b-packed", "-n", "64", "-p", $story)
  Run "q35-smallmode-b" @("run", "models\Qwen3.5-35B-A3B-Q5_K_M.gguf", "--packed", "E:\moe\35b-packed", "-n", "64", "--ram", "4.9", "-p", $story)
  Stop-Process $lim.Id -Force; Start-Sleep 5
}
Get-Process llama-cli, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
"EXIT done"
