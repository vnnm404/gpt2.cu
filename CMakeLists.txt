cmake_minimum_required(VERSION 3.18)
project(gpt2_cuda LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(programs)
add_subdirectory(tests)
