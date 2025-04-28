#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "whisper/utils/gpu_memory_pool.h"

namespace whisper {
namespace beam_search {

struct Token {
    float score;         // Cumulative log probability
    int32_t token_id;    
    int32_t prev_index;  // Index to previous token for history tracking
    int32_t batch_index; // Index of the batch this token belongs to
    
    __host__ __device__ Token() : score(0.0f), token_id(0), prev_index(-1), batch_index(0) {}
    
    __host__ __device__ Token(float s, int32_t t, int32_t p, int32_t b = 0) 
        : score(s), token_id(t), prev_index(p), batch_index(b) {}
};

// For backward compatibility, alias the GPU memory pool
using BeamSearchWorkspace = utils::GPUMemoryPool;

} // namespace beam_search
} // namespace whisper