#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/cuda_utils.h"
#include <stdexcept>

namespace whisper {
namespace beam_search {

// Constructs a workspace with initial capacity in bytes.
BeamSearchWorkspace::BeamSearchWorkspace(std::size_t initialSize)
    : deviceMemory_(nullptr), capacity_(initialSize), usedSize_(0) {
    CUDA_CHECK(cudaMalloc(&deviceMemory_, capacity_));
}

BeamSearchWorkspace::~BeamSearchWorkspace() {
    if (deviceMemory_) {
        cudaFree(deviceMemory_);
        deviceMemory_ = nullptr;
    }
}

// Allocates 'size' bytes with specified alignment; returns device pointer.
void* BeamSearchWorkspace::allocate(std::size_t size, std::size_t alignment) {
    // Align the current offset to the specified boundary
    std::size_t alignedOffset = (usedSize_ + alignment - 1) & ~(alignment - 1);

    if (alignedOffset + size > capacity_) {
        // Double the capacity or increase to fit this allocation
        std::size_t newCapacity = std::max(capacity_ * 2, alignedOffset + size + alignment);
        void* newMemory = nullptr;
        CUDA_CHECK(cudaMalloc(&newMemory, newCapacity));
        if (usedSize_ > 0) {
            CUDA_CHECK(cudaMemcpy(newMemory, deviceMemory_, usedSize_, cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaFree(deviceMemory_));
        deviceMemory_ = newMemory;
        capacity_ = newCapacity;
    }
    // Return aligned pointer within the buffer
    void* result = static_cast<char*>(deviceMemory_) + alignedOffset;
    usedSize_ = alignedOffset + size;
    return result;
}

// Resets the workspace, freeing previous allocations.
void BeamSearchWorkspace::reset() {
    usedSize_ = 0;
}

// Returns number of bytes currently allocated.
std::size_t BeamSearchWorkspace::usedSize() const {
    return usedSize_;
}

// Returns current capacity in bytes.
std::size_t BeamSearchWorkspace::capacity() const {
    return capacity_;
}

} // namespace beam_search
} // namespace whisper 