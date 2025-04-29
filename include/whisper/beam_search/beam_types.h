#pragma once

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include <cstddef>
#include <cstdint>

namespace whisper {
namespace beam_search {

// Holds a token with its score and history pointer for beam search.
struct Token {
    float score;         // Cumulative log probability
    int32_t tokenId;     // Token identifier
    int32_t prevIndex;   // Index of previous token for history tracking
    
    __host__ __device__ Token() : score(0.0f), tokenId(0), prevIndex(-1) {}
    
    __host__ __device__ Token(float s, int32_t t, int32_t p) 
        : score(s), tokenId(t), prevIndex(p) {}
};

// Workspace for GPU memory allocations used in beam search.
class BeamSearchWorkspace {
public:
    // Constructs a workspace with initial capacity in bytes.
    explicit BeamSearchWorkspace(std::size_t initialSize = 16 * 1024 * 1024);
    
    ~BeamSearchWorkspace();
    
    // Allocates 'size' bytes with specified alignment; returns device pointer.
    void* allocate(std::size_t size, std::size_t alignment = 256);
    
    // Resets the workspace, freeing previous allocations.
    void reset();
    
    // Returns number of bytes currently allocated.
    std::size_t usedSize() const;
    
    // Returns current capacity in bytes.
    std::size_t capacity() const;
    
private:
    // Pointer to device memory buffer.
    void* deviceMemory_;
    // Total capacity of the buffer.
    std::size_t capacity_;
    // Bytes used so far.
    std::size_t usedSize_;
    
    // Non-copyable.
    BeamSearchWorkspace(const BeamSearchWorkspace&) = delete;
    BeamSearchWorkspace& operator=(const BeamSearchWorkspace&) = delete;
};

} // namespace beam_search
} // namespace whisper