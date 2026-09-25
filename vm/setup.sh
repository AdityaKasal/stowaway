#!/bin/bash
# Inside the WSL VM (as root): build tools, then copy the project in and build llama.cpp CPU-only.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq build-essential cmake python3-numpy python3-yaml python3-requests git curl >/tmp/apt.log 2>&1
echo "tools: $(gcc --version | head -1) | $(cmake --version | head -1) | numpy $(python3 -c 'import numpy; print(numpy.__version__)')"

SRC=/mnt/c/Users/FSociety/moe-router-study
DST=/root/moe
mkdir -p $DST/models/122b
# code only (the Windows build folder and models are left out)
tar -C $SRC --exclude=./build --exclude=./llama.cpp/build --exclude=./models --exclude=./results --exclude=./.venv \
    --exclude=./logs --exclude=./zips --exclude=./bin -cf - . | tar -C $DST -xf -
cd $DST/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON >/tmp/cmake.log 2>&1
cmake --build build -j 4 --target llama-cli llama-server >/tmp/build.log 2>&1 || { tail -20 /tmp/build.log; exit 1; }
ls -la build/bin/llama-cli build/bin/llama-server | awk '{print $5, $9}'
