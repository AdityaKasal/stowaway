# Builds llama.cpp plus expert-logger. Used on Windows; the Mac uses expert-logger/Makefile.
cmake_minimum_required(VERSION 3.14)
project(moe-router-study C CXX)

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   ON  CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp)

add_executable(expert-logger expert-logger/expert-logger.cpp)
target_link_libraries(expert-logger PRIVATE llama-common llama ggml-cpu)
target_compile_features(expert-logger PRIVATE cxx_std_17)
target_include_directories(expert-logger PRIVATE llama.cpp/common)
set_target_properties(expert-logger PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
