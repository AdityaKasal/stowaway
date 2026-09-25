# waits for the gpt-oss downloads and the Linux build; exactness test; then speed at 4/8/16 GB; restores .wslconfig
$root = "C:\Users\FSociety\moe-router-study"
while (-not ((Get-Content "$root\logs\buildlinux.log" -Tail 1 -ErrorAction SilentlyContinue) -match "^EXIT")) { Start-Sleep 30 }
while (-not (Select-String -Path "$root\logs\gptossdl.log" -Pattern "gpt-oss-20b done" -Quiet)) { Start-Sleep 60 }
"##### exactness (normal VM)"
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/gptoss-exact.sh
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
function Cfg($gb) { Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=$gb`r`nprocessors=4`r`nswap=0"; wsl --shutdown }
"##### 20b at 8 GB"; Cfg "8GB";  wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/gptoss-speed.sh 20b plain fast
"##### 20b at 4 GB"; Cfg "4GB";  wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/gptoss-speed.sh 20b plain
Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force; wsl --shutdown
while (-not (Select-String -Path "$root\logs\gptossdl.log" -Pattern "gpt-oss-120b done" -Quiet)) {
  if (-not (Get-ScheduledTask -TaskName "moe-gptossdl" | Where-Object State -eq "Running")) { powershell -ExecutionPolicy Bypass -File "$root\scripts\run-task.ps1" gptossdl "$root\scripts\gptoss-dl.ps1" }
  Start-Sleep 60 }
"##### pack 120b (normal VM, slim)"
wsl -d Ubuntu -u root -e bash -c "cd /root/moe && python3 repack_experts.py models/gpt-oss/gpt-oss-120b-MXFP4.gguf models/gpt-oss/gpt-oss-120b-MXFP4-experts-packed --slim | grep -vE 'layer +[0-9]+ done'"
"##### 120b at 8 GB"; Cfg "8GB";  wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/gptoss-speed.sh 120b plain fast
"##### 120b at 16 GB"; Cfg "16GB"; wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/gptoss-speed.sh 120b fast
Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force; Remove-Item $env:USERPROFILE\.wslconfig.moe-backup; wsl --shutdown
"wslconfig restored: " + ((Get-Content $env:USERPROFILE\.wslconfig) -join " / ")
"EXIT done"
