Set-Location C:\Users\FSociety\moe-router-study
$D = "C:\Users\FSociety\stream-test"; Remove-Item -Recurse -Force $D -ErrorAction SilentlyContinue
$t0 = Get-Date
& .venv\Scripts\python.exe -c "import moe, streampack; e = moe.CATALOG['gpt-oss-20b']; exp = [moe.hf_checksum(e['repo'], f) for f in e['files']]; streampack.fetch_packed(e['repo'], e['files'], r'$D', r'$D\gpt-oss-20b-MXFP4-experts-packed', exp, 'stowaway-test')" 2>&1 | Select-Object -Last 2
"took $([int]((Get-Date) - $t0).TotalSeconds) s"
(Get-FileHash -Algorithm SHA256 "$D\gpt-oss-20b-MXFP4-experts-packed.bin").Hash.Substring(0,16).ToLower()
& .venv\Scripts\python.exe -c "import sparse; print('model file on disk: %.2f GB of %.2f GB' % (sparse.allocated(r'$D\gpt-oss-20b-MXFP4.gguf')/1e9, __import__('os').path.getsize(r'$D\gpt-oss-20b-MXFP4.gguf')/1e9)); print('packed on disk: %.2f GB' % (sparse.allocated(r'$D\gpt-oss-20b-MXFP4-experts-packed.bin')/1e9))"
"EXIT done"
