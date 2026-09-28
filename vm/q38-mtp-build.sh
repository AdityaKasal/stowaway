#!/bin/bash
# Build llama.cpp + our patch + the Qwen3.8 MTP pull request (ggml-org/llama.cpp#28243) in its own folder
export PATH=/usr/local/bin:$PATH
SRC=/mnt/c/Users/FSociety/moe-router-study
mkdir -p /root/moe-mtp && cd /root/moe-mtp
rsync -a --delete --exclude '/build*/' --exclude '/.git/' --exclude '/models/' --exclude '/docs/' $SRC/llama.cpp/ llama.cpp/
rsync -a $SRC/mtp-files/ llama.cpp/
cmake -S llama.cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" 2>&1 | tail -1
nice -n 10 cmake --build build --target llama-cli llama-server llama-perplexity -j 16 2>&1 | grep -E "error|FAILED" | head -20
mkdir -p $SRC/dist/linux-mtp && cp build/bin/llama-cli build/bin/llama-server build/bin/llama-perplexity $SRC/dist/linux-mtp/ && echo built
