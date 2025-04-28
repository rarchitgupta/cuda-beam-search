#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace whisper {
namespace beam_search {

struct Token {
    float score;         // Cumulative log probability
    int32_t token_id;    
    int32_t prev_index;  // Index to previous token for history tracking
    
    __host__ __device__ Token() : score(0.0f), token_id(0), prev_index(-1) {}
    
    __host__ __device__ Token(float s, int32_t t, int32_t p) 
        : score(s), token_id(t), prev_index(p) {}
};

// Memory manager that avoids repeated GPU memory allocations
class BeamSearchWorkspace {
public:
    BeamSearchWorkspace(size_t initial_size = 16 * 1024 * 1024);
    
    ~BeamSearchWorkspace();
    
    void* Allocate(size_t size, size_t alignment = 256);
    
    void Reset();
    
    size_t GetUsedSize() const;
    
    size_t GetCapacity() const;
    
private:
    void* d_memory_;
    size_t capacity_;
    size_t used_;
    
    BeamSearchWorkspace(const BeamSearchWorkspace&) = delete;
    BeamSearchWorkspace& operator=(const BeamSearchWorkspace&) = delete;
};

} // namespace beam_search
} // namespace whisper