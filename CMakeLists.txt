cmake_minimum_required(VERSION 3.18)
project(gpt2_cu LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA architecture settings - adjust based on your GPU
# Common options: 70, 75, 80, 86, 89, 90
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

# Enable testing
enable_testing()

# Test executables
add_executable(test_mk tests/test_mk.cu)
target_link_libraries(test_mk CUDA::cudart)

add_executable(test_split tests/test_split.cu)
target_link_libraries(test_split CUDA::cudart)

# Add tests
add_test(NAME test_mk COMMAND test_mk)
add_test(NAME test_split COMMAND test_split)
