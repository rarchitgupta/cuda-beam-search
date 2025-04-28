cmake_minimum_required(VERSION 3.18)
project(whisper-cuda-decoder LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)  # For RTX 4060 Ti

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11, and if not found, fetch it
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "pybind11 not found, fetched from GitHub")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}  # Add CUDA include directories
)

# Add source files for the main library
file(GLOB_RECURSE LIB_SOURCES 
    "src/whisper/*.cc"
    "src/whisper/*.cu"
)

# Remove python_bindings.cc from the main library sources
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*python_bindings\\.cc$")

# Create main library
add_library(whisper-cuda-decoder SHARED ${LIB_SOURCES})

# Link CUDA libraries
target_link_libraries(whisper-cuda-decoder
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# Python module
pybind11_add_module(cuda_beam_search src/whisper/beam_search/python_bindings.cc)
target_link_libraries(cuda_beam_search PRIVATE whisper-cuda-decoder)

# Set common output directory for all tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Tests
enable_testing()

# Define test order (basic components first, integration test last)
set(TEST_ORDER
    "test_beam_types"       # Test basic types first
    "test_stream_manager"   # Test stream manager
    "test_tensor_bridge"    # Test tensor bridge
    "test_beam_array"       # Test beam array
    "test_logit_processor"  # Test logit processor
    "test_sequence_manager" # Test sequence manager
    "test_integration"      # Integration test last
)

# Gather all test source files
file(GLOB TEST_SOURCES "tests/test_*.cc")

# Track the test targets to populate TEST_TARGETS variable
set(TEST_TARGETS "")

# Create test targets
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} whisper-cuda-decoder)
    list(APPEND TEST_TARGETS ${test_name})
    
    # Add to ALL target so it's built with make
    set_target_properties(${test_name} PROPERTIES EXCLUDE_FROM_ALL FALSE)
endforeach()

# Add tests in the specified order
foreach(test_name ${TEST_ORDER})
    add_test(NAME ${test_name} COMMAND ${test_name})
    # Set test dependencies to enforce order (each test depends on previous tests)
    if(NOT ${test_name} STREQUAL "test_beam_types")
        list(FIND TEST_ORDER ${test_name} current_index)
        math(EXPR prev_index "${current_index} - 1")
        list(GET TEST_ORDER ${prev_index} prev_test)
        set_tests_properties(${test_name} PROPERTIES DEPENDS ${prev_test})
    endif()
endforeach()

# Custom test target to run all tests in order
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --schedule-random off
    DEPENDS ${TEST_TARGETS}
    COMMENT "Running all tests in order"
)

# Install targets
install(TARGETS whisper-cuda-decoder
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install Python module
install(TARGETS cuda_beam_search
    LIBRARY DESTINATION "${Python_SITEARCH}"
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
) 