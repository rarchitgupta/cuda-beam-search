cmake_minimum_required(VERSION 3.18)
project(whisper-cuda-decoder LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)  # For RTX 4060 Ti

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cc"
    "src/*.cu"
)

# Create library
add_library(whisper-cuda-decoder SHARED ${SOURCES})

# Link CUDA libraries
target_link_libraries(whisper-cuda-decoder
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# Install targets
install(TARGETS whisper-cuda-decoder
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
) 