# waits for the 35B Q8 setup, pauses the 122B download, runs the 8 GB test, restores .wslconfig, resumes the download
$root = "C:\Users\FSociety\moe-router-study"
while (-not (Select-String -Path "$root\logs\q8prep.log" -Pattern "^EXIT" -Quiet)) { Start-Sleep 60 }
Stop-ScheduledTask -TaskName "moe-q8122prep" -ErrorAction SilentlyContinue   # curl -C - resumes it later
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=8GB`r`nprocessors=4`r`nswap=0"
wsl --shutdown
wsl -d Ubuntu -u root -- bash -c "tr -d '\r' < /mnt/c/Users/FSociety/moe-router-study/vm/q8-test.sh > /root/q8-test.sh && bash /root/q8-test.sh"
wsl --shutdown
Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force
Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
"wslconfig restored: " + ((Get-Content $env:USERPROFILE\.wslconfig) -join " / ")
Start-ScheduledTask -TaskName "moe-q8122prep"
"122B download resumed"
