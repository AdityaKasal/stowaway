$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }
$cuda = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
$env:PATH = "$cuda\bin;C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
Set-Location C:\Users\FSociety\moe-router-study
(Get-Process -Id $PID).PriorityClass = "BelowNormal"  # someone may be using the PC; child processes inherit this
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF 2>&1 | Select-Object -Last 2
cmake --build build --target expert-logger llama-bench llama-cli llama-server llama-perplexity -j 8 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 15
