# Compatible Windows engine for CPUs without AVX2 (SSE4.2 only): dist\windows\compat\
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }
$env:PATH = "C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"; $env:CUDA_PATH = ""
Set-Location C:\Users\FSociety\moe-router-study
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
cmake -S llama.cpp -B build-compat -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF `
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF `
  -DGGML_BMI2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_OPENMP=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF `
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON 2>&1 | Select-Object -Last 1
cmake --build build-compat --target llama-cli llama-server -j 8 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 10
New-Item -ItemType Directory -Force dist\windows\compat | Out-Null
Copy-Item build-compat\bin\llama-cli.exe, build-compat\bin\llama-server.exe dist\windows\compat\
Get-ChildItem dist\windows\compat | Select-Object Name, Length | Format-Table | Out-String
"EXIT done"
