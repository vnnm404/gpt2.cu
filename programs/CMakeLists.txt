# Build programs

# Find all CUDA source files in the current directory
file(GLOB PROGRAM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# Create an executable for each source file
foreach(PROGRAM_SOURCE ${PROGRAM_SOURCES})
    # Get the filename without extension
    get_filename_component(PROGRAM_NAME ${PROGRAM_SOURCE} NAME_WE)
    
    # Create executable
    add_executable(${PROGRAM_NAME} ${PROGRAM_SOURCE})
    
    # inference.cu includes layer sources directly, others link with gpt2 library
    if(NOT PROGRAM_NAME STREQUAL "inference")
        target_link_libraries(${PROGRAM_NAME} gpt2)
    endif()
    
    # Set CUDA properties
    set_source_files_properties(${PROGRAM_SOURCE} PROPERTIES LANGUAGE CUDA)
    set_target_properties(${PROGRAM_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
