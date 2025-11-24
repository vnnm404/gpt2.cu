# Build programs

# Training executable
add_executable(train train.c)
target_link_libraries(train gpt2)
set_target_properties(train PROPERTIES LINKER_LANGUAGE C)

# Inference executable
add_executable(inference inference.c)
target_link_libraries(inference gpt2)
set_target_properties(inference PROPERTIES LINKER_LANGUAGE C)

# Benchmark executable
add_executable(benchmark benchmark.c)
target_link_libraries(benchmark gpt2)
set_target_properties(benchmark PROPERTIES LINKER_LANGUAGE C)
