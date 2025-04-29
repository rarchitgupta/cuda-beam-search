#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace whisper {
namespace beam_search {

// GPU kernel to prune each beam to its top-K scoring tokens using CUB segmented sort.
void launchPruneKernel(
    float* scores,
    int* tokenIds,
    int* prevIndices,
    int beamCount,
    int candidateCount,
    int beamWidth,
    cudaStream_t stream = 0
);

} // namespace beam_search
} // namespace whisper 