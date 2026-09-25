Set-Location C:\Users\FSociety\moe-router-study
$env:SPEC_TAG = "sp6-code";  $env:SPEC_PROMPT = "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\spec-test6.ps1
$env:SPEC_TAG = "sp6-story"; $env:SPEC_PROMPT = "Write the opening two sentences of a mystery story set in a lighthouse."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\spec-test6.ps1
