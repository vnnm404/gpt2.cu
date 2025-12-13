# Build Unity
add_library(unity STATIC ${CMAKE_SOURCE_DIR}/third_party/unity.c)
target_include_directories(unity PUBLIC ${CMAKE_SOURCE_DIR}/third_party)

# Collect CUDA test files
file(GLOB TEST_SOURCES "test_*.cu")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # IMPORTANT: tell CMake these are CUDA sources
    set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE CUDA)

    add_executable(${TEST_NAME} ${TEST_SOURCE})

    # Includes
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/third_party)

    # Link everything
    target_link_libraries(${TEST_NAME} gpt2 unity)

    # DO NOT FORCE LINKER LANGUAGE TO C!
    # This was the cause of your kernel differences.

    # Enable proper CUDA device linking (MUST MATCH PROGRAMS)
    set_target_properties(${TEST_NAME} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endforeach()

# lib w/ extern entrypoints
add_library(mk_entrypoint SHARED test_train_mk.cu)
set_target_properties(mk_entrypoint PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mk_entrypoint PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(mk_entrypoint gpt2 unity)


# nvbench ----------------------------------------------------------------------
include(FetchContent)

option(GPT2_ENABLE_NVBENCH "Build NVBench benchmarks" ON)

if (GPT2_ENABLE_NVBENCH)
  FetchContent_Declare(
    nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    # TODO: pin
    GIT_TAG main
  )
  FetchContent_MakeAvailable(nvbench)
endif()

if (GPT2_ENABLE_NVBENCH)
  file(GLOB BENCH_SOURCES "bench_*.cu")

  foreach(BENCH_SOURCE ${BENCH_SOURCES})
    get_filename_component(BENCH_NAME ${BENCH_SOURCE} NAME_WE)
    set_source_files_properties(${BENCH_SOURCE} PROPERTIES LANGUAGE CUDA)

    add_executable(${BENCH_NAME} ${BENCH_SOURCE})

    target_include_directories(${BENCH_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
    target_link_libraries(${BENCH_NAME} gpt2 nvbench::main)

    set_target_properties(${BENCH_NAME} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
  endforeach()
endif()

