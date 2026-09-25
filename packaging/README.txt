stowaway - run big AI models on an ordinary computer
====================================================

No graphics card needed. 8 GB of RAM is enough; 4 GB works too, more slowly. The model lives on your SSD
and stowaway loads only the parts it needs for each word.

HOW TO USE
  Double-click "stowaway" (stowaway.exe on Windows). Pick a model. It shows
  the download size and asks before downloading anything. After a one-time
  setup, a chat opens in your web browser. Keep the stowaway window open
  while you chat; closing it turns the AI off.

  Or from a terminal:
    stowaway list                      models it can download
    stowaway run qwen3.5-35b           download (asks first), set up, chat in the browser
    stowaway run qwen3.5-35b --cli     chat in the terminal instead
    stowaway run qwen3.5-35b --fast    faster answers and first word; answers differ slightly
    stowaway run qwen3.5-35b --think   let the model think before answering (slower)
    stowaway plan qwen3.5-35b          show the memory plan and expected speed
    stowaway run path/to/model.gguf    any other Mixture-of-Experts GGUF model

WHAT TO EXPECT (8 GB RAM, no GPU, normal laptop SSD)
  qwen3.5-35b    26 GB download, ~5-8 words per second
                 (on a 4 GB machine: ~2 words per second)
  qwen3.5-122b   92 GB download, ~1 word per second

  You need about the download size in free disk space, plus 10%.
  An SSD is required; on a hard drive it will be far too slow.
  Models are saved in a "moe-models" folder in your home folder. Set the
  MOE_HOME environment variable to put them somewhere else.

OLDER PROCESSORS
  On Intel/AMD processors without AVX2 (most PCs before 2013, many budget Celerons and
  Pentiums), stowaway automatically uses the engine in the "compat" folder: it works
  everywhere, just slower. Keep that folder next to the program.

UPDATES
  At start, stowaway asks GitHub whether a newer version exists and says so.
  Set STOWAWAY_NO_UPDATE_CHECK=1 to turn that off. "stowaway --version" shows yours.

FIRST RUN WARNINGS
  Windows: "Windows protected your PC" -> More info -> Run anyway.
  Mac: if it says the app can't be opened, run this once in Terminal from
       this folder:  xattr -dr com.apple.quarantine .
       (stowaway-mac is for Apple Silicon Macs, stowaway-mac-intel for Intel Macs.)
  Linux: needs glibc 2.34+ (Ubuntu 22.04+, Debian 12+, Fedora 35+).
