# Rebuild the release llama-cli / llama-server (not llama-perplexity, which a sweep may be using) and stowaway.exe
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }
$env:PATH = "C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
Set-Location C:\Users\FSociety\moe-router-study
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
cmake --build build-dist --target llama-cli llama-server -j 8 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 10
& .venv\Scripts\python.exe -m PyInstaller --onefile --name stowaway --paths llama.cpp\gguf-py --distpath dist\windows `
  --workpath build-pyi --specpath build-pyi --noconfirm --exclude-module tkinter --exclude-module torch `
  --exclude-module sentencepiece moe.py 2>&1 | Select-Object -Last 1
Copy-Item build-dist\bin\llama-cli.exe, build-dist\bin\llama-server.exe dist\windows\
Get-ChildItem dist\windows | Select-Object Name, Length, LastWriteTime | Format-Table | Out-String
"EXIT done"
