cmake_minimum_required(VERSION 3.18)
project(whisper-cuda-decoder LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)  # For RTX 4060 Ti

# Find Python and pybind11
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

# If pybind11 wasn't found through find_package, fetch it
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
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