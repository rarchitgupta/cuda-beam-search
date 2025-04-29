#include "whisper/beam_search/history_reconstruct.h"
#include <cuda_runtime.h>

namespace whisper {
namespace beam_search {

__global__ void reconstruct_kernel(
    const int* historyPrev,
    const int* historyTokens,
    std::size_t beamWidth,
    std::size_t stepCount,
    int* outputSequences)
{
    int beam = blockIdx.x * blockDim.x + threadIdx.x;
    if (beam >= beamWidth) return;
    // Start from the last step and this beam
    int idx = beam;
    // Write tokens in reverse, then flip
    for (int step = stepCount; step >= 0; --step) {
        int token = historyTokens[step * beamWidth + idx];
        outputSequences[beam * (stepCount + 1) + step] = token;
        idx = historyPrev[step * beamWidth + idx];
        if (idx < 0) break;
    }
}

void launchReconstructKernel(
    const int* d_historyPrevIndices,
    const int* d_historyTokenIds,
    std::size_t beamWidth,
    std::size_t stepCount,
    int* d_outputSequences,
    cudaStream_t stream)
{
    int threads = 32;
    int blocks = (beamWidth + threads - 1) / threads;
    reconstruct_kernel<<<blocks, threads, 0, stream>>>(
        d_historyPrevIndices, d_historyTokenIds, beamWidth, stepCount, d_outputSequences);
}

} // namespace beam_search
} // namespace whisper 