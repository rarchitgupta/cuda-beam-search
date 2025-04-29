cmake_minimum_required(VERSION 3.18)
project(whisper-cuda-decoder LANGUAGES CXX CUDA)

# After project declaration, enable separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

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
    # Exclude CUDA sources under beam_search (collected in beam_kernels)
    # Use only any .cu outside beam_search, if needed
    "src/whisper/*.cu"
)
# Remove any CUDA sources in beam_search to avoid duplicate definitions
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*/beam_search/.*\\.cu$")

# Remove python_bindings.cc from the main library sources
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*python_bindings\\.cc$")

# Build a static library of beam search CUDA kernels for device linking
file(GLOB BEAM_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/whisper/beam_search/*.cu"
)
add_library(beam_kernels STATIC ${BEAM_KERNEL_SOURCES})
# Enable separable compilation, PIC, and resolve device symbols
set_target_properties(beam_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Create main library
add_library(whisper-cuda-decoder SHARED ${LIB_SOURCES})
set_target_properties(whisper-cuda-decoder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA libraries and the beam_kernels static library
target_link_libraries(whisper-cuda-decoder PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    beam_kernels
)

# Python module
pybind11_add_module(cuda_beam_search src/whisper/beam_search/python_bindings.cc)
target_link_libraries(cuda_beam_search PRIVATE whisper-cuda-decoder)

# Set common output directory for all tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Tests
enable_testing()

# Gather all test source files (including our new smoke test)
file(GLOB TEST_SOURCES "tests/test_*.cc" "tests/test_process_logits.cc")

# Create test targets
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    # For test_argmax, compile as CUDA to allow kernel launch syntax
    if(test_name STREQUAL "test_argmax")
        set_source_files_properties(${test_source} PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${test_name} ${test_source})
    # Link the main library and beam_kernels static library
    target_link_libraries(${test_name} PRIVATE
        whisper-cuda-decoder
        beam_kernels
    )
    # Ensure separable CUDA device code is linked
    set_target_properties(${test_name} PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        BUILD_RPATH "\$ORIGIN/.."
        INSTALL_RPATH "\$ORIGIN/.."
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
    
    # Add to ALL target so it's built with make
    set_target_properties(${test_name} PROPERTIES EXCLUDE_FROM_ALL FALSE)
endforeach()

# Custom test target to run all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_TARGETS}
    COMMENT "Running all tests"
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