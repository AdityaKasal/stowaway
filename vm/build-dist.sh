#!/bin/bash
# Standalone Linux x64 build of moe on Ubuntu 22.04 (glibc 2.35): CPU-only llama.cpp, static libstdc++, plus moe.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential python3-venv python3-pip rsync curl binutils > /dev/null
if ! cmake --version 2>/dev/null | grep -q "version [34]\.[3-9][0-9]"; then python3 -m pip install -q cmake ninja; fi
cmake --version | head -1
SRC=/mnt/c/Users/FSociety/moe-router-study
mkdir -p ~/moe && cd ~/moe
rsync -a --delete --exclude '/build/' --exclude '/build-*/' --exclude '/.git/' --exclude '/models/' $SRC/llama.cpp/ llama.cpp/
cp $SRC/moe.py $SRC/repack_experts.py $SRC/pack_dense.py $SRC/sparse.py .
cmake -S llama.cpp -B build-dist -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" | tail -2
cmake --build build-dist --target llama-cli llama-server -j 16 2>&1 | grep -E "error|FAILED" | head -15 || true
[ -d venv ] || python3 -m venv venv
venv/bin/pip install -q numpy pyyaml pyinstaller
venv/bin/python -m PyInstaller --onefile --name stowaway --paths llama.cpp/gguf-py --distpath dist/linux --workpath build-pyi \
  --specpath build-pyi --noconfirm --exclude-module tkinter --exclude-module torch --exclude-module sentencepiece moe.py 2>&1 | tail -1
cp build-dist/bin/llama-cli build-dist/bin/llama-server dist/linux/
ls -la dist/linux
ldd dist/linux/llama-server
objdump -T dist/linux/llama-server dist/linux/stowaway | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1
mkdir -p $SRC/dist/linux && cp dist/linux/* $SRC/dist/linux/
echo EXIT done
