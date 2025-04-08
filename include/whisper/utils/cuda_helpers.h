#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace whisper {
namespace utils {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + \
                cudaGetErrorString(error)); \
        } \
    } while(0)

// Memory management class
class GPUMemoryPool {
public:
    GPUMemoryPool(size_t initial_size = 0);
    ~GPUMemoryPool();

    // Allocate memory on GPU
    void* Allocate(size_t size);

    // Free memory on GPU
    void Free(void* ptr);

    // Preallocate a pool of memory
    void Preallocate(size_t size);

private:
    void* pool_;
    size_t total_allocated_;
    size_t max_memory_;
};

} // namespace utils
} // namespace whisper 