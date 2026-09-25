LLAMA := ../llama.cpp
LIBS  := $(LLAMA)/build/bin

expert-logger: expert-logger.cpp ../llama.cpp/common/moe-stream.h
	clang++ -std=c++17 -O2 -o $@ $< \
		-I$(LLAMA)/include -I$(LLAMA)/common -I$(LLAMA)/ggml/include \
		-L$(LIBS) -lllama-common -lllama -lggml -lggml-base -lggml-cpu \
		-Wl,-rpath,$(abspath $(LIBS))
