#include "whisper/utils/gpu_memory_pool.h"
#include <stdexcept>
#include <string>
#include <algorithm>

namespace whisper {
namespace utils {

GPUMemoryPool::GPUMemoryPool(size_t initial_size) 
    : d_memory_(nullptr), capacity_(initial_size), used_(0) {
    CUDA_CHECK(cudaMalloc(&d_memory_, capacity_), "Failed to allocate GPU memory pool");
}

GPUMemoryPool::~GPUMemoryPool() {
    if (d_memory_) {
        cudaFree(d_memory_);
        d_memory_ = nullptr;
    }
}

void* GPUMemoryPool::Allocate(size_t size, size_t alignment) {
    // Align the current offset to the specified boundary
    size_t aligned_offset = (used_ + alignment - 1) & ~(alignment - 1);
    
    if (aligned_offset + size > capacity_) {
        // Double the capacity or increase to fit this allocation
        size_t new_capacity = std::max(capacity_ * 2, aligned_offset + size + alignment);
        void* new_memory = nullptr;
        
        CUDA_CHECK(cudaMalloc(&new_memory, new_capacity), "Failed to reallocate GPU memory pool");
        
        if (used_ > 0) {
            CUDA_CHECK(cudaMemcpy(new_memory, d_memory_, used_, cudaMemcpyDeviceToDevice), 
                      "Failed to copy data during GPU memory pool resize");
        }
        
        CUDA_CHECK(cudaFree(d_memory_), "Failed to free old GPU memory during resize");
        d_memory_ = new_memory;
        capacity_ = new_capacity;
    }
    
    void* result = static_cast<char*>(d_memory_) + aligned_offset;
    used_ = aligned_offset + size;
    return result;
}

void GPUMemoryPool::Free(void* ptr) {
    // No-op in pooled implementation
}

void GPUMemoryPool::Reset() {
    used_ = 0;
}

void GPUMemoryPool::EnsureCapacity(size_t size) {
    if (size > capacity_) {
        void* new_memory = nullptr;
        
        CUDA_CHECK(cudaMalloc(&new_memory, size), "Failed to allocate GPU memory in EnsureCapacity");
        
        if (used_ > 0) {
            CUDA_CHECK(cudaMemcpy(new_memory, d_memory_, used_, cudaMemcpyDeviceToDevice),
                      "Failed to copy data during GPU memory pool expansion");
        }
        
        CUDA_CHECK(cudaFree(d_memory_), "Failed to free old GPU memory during expansion");
        d_memory_ = new_memory;
        capacity_ = size;
    }
}

size_t GPUMemoryPool::GetUsedSize() const {
    return used_;
}

size_t GPUMemoryPool::GetCapacity() const {
    return capacity_;
}

} // namespace utils
} // namespace whisper 