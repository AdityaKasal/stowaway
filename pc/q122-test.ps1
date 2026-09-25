# after the 122B Q8 download+pack: slim it, then the 8 GB / 3 GB/s test; restores .wslconfig
$root = "C:\Users\FSociety\moe-router-study"
$slim = wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q122-slim.sh 2>&1
$slim
if (-not ($slim -match "SLIM-OK")) { "packing failed; not running the test"; "EXIT done"; exit 1 }
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=8GB`r`nprocessors=4`r`nswap=0"
wsl --shutdown
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/q122-test.sh
wsl --shutdown
Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force
Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
"wslconfig restored: " + ((Get-Content $env:USERPROFILE\.wslconfig) -join " / ")
"EXIT done"
