# Standalone Windows build of moe: CPU-only llama.cpp with static runtime (no DLLs, no CUDA, no OpenMP), plus moe.exe.
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }
$env:PATH = "C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
$env:CUDA_PATH = ""
Set-Location C:\Users\FSociety\moe-router-study
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
cmake -S llama.cpp -B build-dist -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF `
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON `
  -DGGML_OPENMP=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF `
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON 2>&1 | Select-Object -Last 2
cmake --build build-dist --target llama-cli llama-server -j 8 2>&1 | Select-String -Pattern "error|FAILED" | Select-Object -First 15
& .venv\Scripts\python.exe -m pip install -q pyinstaller 2>&1 | Select-Object -Last 2
& .venv\Scripts\python.exe -m PyInstaller --onefile --name moe --paths llama.cpp\gguf-py --distpath dist\windows `
  --workpath build-pyi --specpath build-pyi --noconfirm --exclude-module tkinter --exclude-module torch `
  --exclude-module sentencepiece moe.py 2>&1 | Select-Object -Last 1
Copy-Item build-dist\bin\llama-cli.exe, build-dist\bin\llama-server.exe dist\windows\
Get-ChildItem dist\windows | Select-Object Name, Length
& dumpbin /dependents dist\windows\llama-server.exe | Select-String "\.dll"
"EXIT done"
