#!/bin/bash
# Compatible Linux engine for CPUs without AVX2 (SSE4.2 only): dist/linux/compat/
set -e
export PATH=/usr/local/bin:$PATH
cd ~/moe
rsync -a --delete --exclude '/build/' --exclude '/build-*/' --exclude '/.git/' --exclude '/models/' /mnt/c/Users/FSociety/moe-router-study/llama.cpp/ llama.cpp/
cmake -S llama.cpp -B build-compat -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_BMI2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" | tail -1
cmake --build build-compat --target llama-cli llama-server -j 16 2>&1 | grep -E "error|FAILED" | head -10 || true
mkdir -p dist/linux/compat /mnt/c/Users/FSociety/moe-router-study/dist/linux/compat
cp build-compat/bin/llama-cli build-compat/bin/llama-server dist/linux/compat/
cp dist/linux/compat/* /mnt/c/Users/FSociety/moe-router-study/dist/linux/compat/
echo "AVX (ymm) instructions in the compat build: $(objdump -d --no-show-raw-insn dist/linux/compat/llama-cli | grep -c '%ymm')"
echo "AVX (ymm) instructions in the normal build: $(objdump -d --no-show-raw-insn dist/linux/llama-cli | grep -c '%ymm')"
echo EXIT done
