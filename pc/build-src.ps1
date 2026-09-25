# Build llama.cpp (CUDA, RTX 50-series = sm_120) + expert-logger from source.
$root = "C:\Users\FSociety\moe-router-study"
while ((Get-ScheduledTask moe-tools).State -eq "Running") { Start-Sleep 30 }
"tools done $(Get-Date -Format T)"

# load the MSVC compiler environment
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -property installationPath
"VS at $vs"
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
  if ($_ -match "^([^=]+)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] }
}
# freshly installed tools aren't on this session's PATH yet
$cuda = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
$env:CUDA_PATH = $cuda
$env:PATH = "$cuda\bin;C:\Program Files\CMake\bin;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:PATH"
"CUDA at $cuda"
nvcc --version | Select-Object -Last 2
cmake --version | Select-Object -First 1
ninja --version

Set-Location $root
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF 2>&1 | Select-Object -Last 5
cmake --build build --target expert-logger -j 20 2>&1 | Select-Object -Last 15
if ($LASTEXITCODE) { "BUILD FAILED"; exit 1 }
Get-ChildItem build\bin | Select-Object Name, Length
"build done $(Get-Date -Format T)"
