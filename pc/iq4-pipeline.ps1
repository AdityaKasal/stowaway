$games = "Lunar|javaw|Minecraft|Roblox|Spider|Valorant|League|FortniteClient|GTA|RocketLeague"
if (Get-Process | Where-Object { $_.ProcessName -match $games }) { "a game is running: not starting"; "EXIT done"; exit 0 }
New-Item -ItemType Directory -Force C:\Users\FSociety\moe-router-study\results | Out-Null
wsl -d Ubuntu -u root -e bash /mnt/c/Users/FSociety/moe-router-study/vm/iq4-pipeline.sh
"##### Q5 vs Q8 (Windows, the Q5 files on C:)"
Set-Location C:\Users\FSociety\moe-router-study
$env:LLAMA_NO_MMAP_PREFETCH = "1"; $env:MOE_CACHE_GB = "12"; $env:MOE_IO_THREADS = "6"; $env:MOE_STATS = "1"
$env:EXPERT_CACHE_PACKED = "models\122b\experts-packed"; $env:EXPERT_CACHE_CHUNK_KB = "8192"; $env:CUDA_VISIBLE_DEVICES = "-1"
$a = @("-m", "models\122b\Qwen3.5-122B-A10B-Q5_K_M-00001-of-00003.gguf", "-ngl", "0", "--no-repack", "--no-op-offload", "-t", "12",
       "-c", "512", "-b", "512", "-ub", "256", "--chunks", "4", "-f", "data\wikitext\wikitext-2-raw\wiki.test.raw", "--no-warmup",
       "--kl-divergence-base", "results\base-q8-122b.kld", "--kl-divergence")
$p = Start-Process build-dist\bin\llama-perplexity.exe -ArgumentList $a -NoNewWindow -Wait -PassThru -RedirectStandardOutput results\q5-vs-q8.out -RedirectStandardError results\q5-vs-q8.err
Get-Content results\q5-vs-q8.out, results\q5-vs-q8.err | Select-String "Final estimate|Mean    KLD|Same top p:|99.0%   KLD" | ForEach-Object { "  " + $_.Line.Trim() }
"EXIT done"
