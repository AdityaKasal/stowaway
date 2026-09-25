$winget = "$env:LOCALAPPDATA\Microsoft\WindowsApps\winget.exe"
if (-not (Test-Path $winget)) { $winget = (Get-Command winget).Source }
"using $winget"
$common = @("--accept-source-agreements", "--accept-package-agreements", "--disable-interactivity", "--silent", "-e")
function Install($id, $extra) {
  "=== $id  $(Get-Date -Format T)"
  & $winget install --id $id @common @extra
  "winget exit $LASTEXITCODE"
}
Install "Kitware.CMake" @()
Install "Ninja-build.Ninja" @()
# C++ compiler (MSVC) + Windows SDK, no IDE
Install "Microsoft.VisualStudio.2022.BuildTools" @("--override", "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended")
Install "Nvidia.CUDA" @("--version", "13.4")
"=== done $(Get-Date -Format T)"
