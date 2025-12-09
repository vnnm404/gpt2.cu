cmake_minimum_required(VERSION 3.18)
project(gpt2_cuda LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force Release build type with optimizations
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Optimization flags for C/CXX code
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -march=native")

# Optimization flags for CUDA code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG --use_fast_math -Wno-deprecated-gpu-targets -DPROFILE")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# compile_commands.json at repo root
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  add_custom_target(copy_compile_commands ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_SOURCE_DIR}/compile_commands.json
        BYPRODUCTS ${CMAKE_SOURCE_DIR}/compile_commands.json
        COMMENT "Copy compile_commands.json to source dir")
endif()
