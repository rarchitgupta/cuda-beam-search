#include "whisper/beam_search/beam_types.h"
#include <stdexcept>
#include <string>

namespace whisper {
namespace beam_search {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

BeamSearchWorkspace::BeamSearchWorkspace(size_t initial_size) 
    : d_memory_(nullptr), capacity_(initial_size), used_(0) {
    CUDA_CHECK(cudaMalloc(&d_memory_, capacity_));
}

BeamSearchWorkspace::~BeamSearchWorkspace() {
    if (d_memory_) {
        cudaFree(d_memory_);
        d_memory_ = nullptr;
    }
}

void* BeamSearchWorkspace::Allocate(size_t size, size_t alignment) {
    // Align the current offset to the specified boundary
    size_t aligned_offset = (used_ + alignment - 1) & ~(alignment - 1);
    
    if (aligned_offset + size > capacity_) {
        // Double the capacity or increase to fit this allocation
        size_t new_capacity = std::max(capacity_ * 2, aligned_offset + size + alignment);
        void* new_memory = nullptr;
        
        CUDA_CHECK(cudaMalloc(&new_memory, new_capacity));
        
        if (used_ > 0) {
            CUDA_CHECK(cudaMemcpy(new_memory, d_memory_, used_, cudaMemcpyDeviceToDevice));
        }
        
        CUDA_CHECK(cudaFree(d_memory_));
        d_memory_ = new_memory;
        capacity_ = new_capacity;
    }
    
    void* result = static_cast<char*>(d_memory_) + aligned_offset;
    used_ = aligned_offset + size;
    return result;
}

void BeamSearchWorkspace::Reset() {
    used_ = 0;
}

size_t BeamSearchWorkspace::GetUsedSize() const {
    return used_;
}

size_t BeamSearchWorkspace::GetCapacity() const {
    return capacity_;
}

} // namespace beam_search
} // namespace whisper 