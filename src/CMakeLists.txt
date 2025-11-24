# Build GPT-2 library

file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/layers/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu"
)

add_library(gpt2 STATIC ${SOURCES})
target_include_directories(gpt2 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(gpt2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
set_target_properties(gpt2 PROPERTIES LINKER_LANGUAGE CXX)
