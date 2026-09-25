# Runs a script as a detached scheduled task so it survives SSH disconnects.
# usage: run-task.ps1 <name> <script.ps1>   (output goes to <root>\logs\<name>.log)
param([string]$Name, [string]$Script)
$ErrorActionPreference = "Stop"
$root = "C:\Users\FSociety\moe-router-study"
New-Item -ItemType Directory -Force "$root\logs" | Out-Null
$log = "$root\logs\$Name.log"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"& '$Script' *> '$log'; 'EXIT ' + `$LASTEXITCODE | Add-Content '$log'`"" `
  -WorkingDirectory $root
$principal = New-ScheduledTaskPrincipal -UserId "FSOCIETY\fsociety" -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 6) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Unregister-ScheduledTask -TaskName "moe-$Name" -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName "moe-$Name" -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName "moe-$Name"
"started moe-$Name -> $log"
