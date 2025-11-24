cmake_minimum_required(VERSION 3.18)
project(gpt2_cuda LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug or Release)" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler flags configuration
# Note: We use target properties to avoid generator expression issues

# Debug mode configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling debug symbols and sanitizers")
    
    # Set flags for C code
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -Wall -Wextra -Wpedantic")
    
    # Set flags for CUDA code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0")
    
    # Enable AddressSanitizer and UndefinedBehaviorSanitizer for C code
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined")
    endif()
    
# Release mode configuration
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling optimizations for release")
    
    # Set flags for C code
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG -march=native")
    
    # Set flags for CUDA code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG --use_fast_math")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(programs)
add_subdirectory(tests)
