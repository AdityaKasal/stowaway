# waits for the 35B Q8 prep to finish so the two downloads don't share the connection
$log = "C:\Users\FSociety\moe-router-study\logs\q8prep.log"
while (-not (Select-String -Path $log -Pattern "^EXIT" -Quiet)) { Start-Sleep 60 }
wsl -d Ubuntu -u root -- bash -c "tr -d '\r' < /mnt/c/Users/FSociety/moe-router-study/vm/q8-122-prep.sh > /root/q8-122-prep.sh && bash /root/q8-122-prep.sh"
