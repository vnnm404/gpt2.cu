cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")
endif()
project(gpt2_cuda LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CUTLASS_ROOT "${CMAKE_SOURCE_DIR}/build/cutlass" CACHE PATH "CUTLASS 3.5.1 checkout")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  include(FetchContent)
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG f7b19de32c5d1f3cedfc735c2849f12b537522ee
    SOURCE_DIR "${CUTLASS_ROOT}")
  FetchContent_Populate(cutlass)
endif()

add_library(gpt2_executor SHARED src/executor.cu)
if(CMAKE_CUDA_ARCHITECTURES MATCHES "90a")
  target_compile_definitions(gpt2_executor PRIVATE GPT2_HOPPER=1)
  target_link_libraries(gpt2_executor PRIVATE cuda)
endif()
target_include_directories(gpt2_executor PRIVATE include "${CUTLASS_ROOT}/include")
target_compile_options(gpt2_executor PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>)
# One translation unit keeps the scheduler and all device operations visible
# to the compiler; there is no relocatable device-code linking boundary.
set_target_properties(gpt2_executor PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
