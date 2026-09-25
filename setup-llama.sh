#!/bin/sh
# Get llama.cpp at the commit moe was built against and apply moe's changes (the expert cache and its hooks).
set -e
cd "$(dirname "$0")"
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout d2e54583c7452353eb35d40431281f6ee984332f
git apply ../patches/moe-stream.patch
echo "llama.cpp is ready: build it with cmake (see README.md)"
