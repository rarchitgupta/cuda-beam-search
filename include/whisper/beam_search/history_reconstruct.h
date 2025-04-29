#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace whisper {
namespace beam_search {
// Launches a kernel to reconstruct the best path for each beam using device history arrays.
// d_historyPrevIndices, d_historyTokenIds: [stepCount * beamWidth]
// d_outputSequences: [beamWidth * (stepCount+1)], row-major (one row per beam)
void launchReconstructKernel(
    const int* d_historyPrevIndices,
    const int* d_historyTokenIds,
    std::size_t beamWidth,
    std::size_t stepCount,
    int* d_outputSequences,
    cudaStream_t stream = 0);
} // namespace beam_search
} // namespace whisper 