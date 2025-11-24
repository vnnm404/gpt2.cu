# Build tests with Unity framework

# Build Unity as a library
add_library(unity STATIC ${CMAKE_SOURCE_DIR}/third_party/unity.c)
target_include_directories(unity PUBLIC ${CMAKE_SOURCE_DIR}/third_party)

# Automatically find all test_*.c files
file(GLOB TEST_SOURCES "test_*.c")

# Create an executable for each test file
foreach(TEST_SOURCE ${TEST_SOURCES})
    # Get the filename without extension
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Create executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
    target_link_libraries(${TEST_NAME} gpt2 unity)
    set_target_properties(${TEST_NAME} PROPERTIES LINKER_LANGUAGE C)
endforeach()
