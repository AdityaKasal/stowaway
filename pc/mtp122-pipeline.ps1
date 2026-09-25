# After the 122B MTP download: pack experts and dense weights, copy into the VM, run the capped VM test.
Set-Location C:\Users\FSociety\moe-router-study
while (-not (Select-String -Path logs\dlmtp.log -Pattern "122B ready" -Quiet)) { Start-Sleep 30 }
$m = "D:\moe\122b-mtp\Qwen3.5-122B-A10B-UD-Q5_K_M-00001-of-00003.gguf"
.\.venv\Scripts\python model_info.py $m
.\.venv\Scripts\python -u repack_experts.py $m D:\moe\122b-mtp-experts-packed 2>&1 | Select-Object -Last 2
.\.venv\Scripts\python pack_dense.py $m D:\moe\122b-mtp-dense-packed
wsl -d Ubuntu -u root -- bash /mnt/c/Users/FSociety/moe-router-study/vm/prep-mtp122.sh
$env:MTP_SCRIPT = "test-mtp122.sh"
$env:MTP_ARGS = "m122 models/122b-mtp/Qwen3.5-122B-A10B-UD-Q5_K_M-00001-of-00003.gguf models/122b-mtp/122b-mtp-experts-packed 0.5 4.0 4.6"
& scripts\vm-test-mtp.ps1
