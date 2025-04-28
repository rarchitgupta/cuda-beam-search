#pragma once

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace whisper {
namespace utils {

// Unified CUDA error checking macro
#define CUDA_CHECK(call, msg) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::string err = std::string(msg) + " at " + __FILE__ + ":" + \
                          std::to_string(__LINE__) + ": " + \
                          cudaGetErrorString(error); \
        throw std::runtime_error(err); \
    } \
} while(0)

// Simplified version without additional message
#define CUDA_CHECK_SIMPLE(call) CUDA_CHECK(call, "CUDA error")

} // namespace utils
} // namespace whisper 