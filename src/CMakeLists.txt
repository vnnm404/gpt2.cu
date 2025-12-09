# Build GPT-2 library

file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu"
)

# Exclude layer files since mk.cu includes them directly for optimization
list(FILTER SOURCES EXCLUDE REGEX ".*/layers/.*\.cu$")

add_library(gpt2 STATIC ${SOURCES})
target_include_directories(gpt2 PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(gpt2 PROPERTIES 
    LINKER_LANGUAGE CXX
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
