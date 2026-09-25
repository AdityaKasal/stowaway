Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=4GB`r`nprocessors=4`r`nswap=0"
wsl --shutdown
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/gptoss-speed.sh 120b plain
wsl --shutdown
Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force
Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
"wslconfig restored: " + ((Get-Content $env:USERPROFILE\.wslconfig) -join " / ")
"EXIT done"
