# 8 GB: 122B split sweep; 4 GB: 35B. Restores .wslconfig afterwards.
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.moe-backup -Force
foreach ($cfg in @(@("8GB", "split-test.sh"), @("4GB", "ram4-test.sh"))) {
  "##### memory=$($cfg[0]) $(Get-Date -Format T)"
  Set-Content $env:USERPROFILE\.wslconfig "[wsl2]`r`nnetworkingMode=mirrored`r`nmemory=$($cfg[0])`r`nprocessors=4`r`nswap=0"
  wsl --shutdown
  wsl -d Ubuntu -u root -e bash "/mnt/c/Users/FSociety/moe-router-study/vm/$($cfg[1])"
}
wsl --shutdown
Copy-Item $env:USERPROFILE\.wslconfig.moe-backup $env:USERPROFILE\.wslconfig -Force
Remove-Item $env:USERPROFILE\.wslconfig.moe-backup
"wslconfig restored: " + ((Get-Content $env:USERPROFILE\.wslconfig) -join " / ")
"EXIT done"
