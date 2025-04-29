#include "whisper/beam_search/prune_kernel.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace whisper {
namespace beam_search {

__global__ void copyTopK(
    const float* in_scores, const int* in_tokenIds, const int* in_prevIndices,
    float* out_scores, int* out_tokenIds, int* out_prevIndices,
    int beamCount, int candidateCount, int beamWidth)
{
    int beam = blockIdx.y;
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    if (beam < beamCount && idx < beamWidth) {
        int in_base  = beam * candidateCount;
        int out_base = beam * candidateCount;
        out_scores   [out_base + idx] = in_scores   [in_base + idx];
        out_tokenIds [out_base + idx] = in_tokenIds [in_base + idx];
        out_prevIndices[out_base + idx] = in_prevIndices[in_base + idx];
    }
}

void launchPruneKernel(
    float* scores,
    int* tokenIds,
    int* prevIndices,
    int beamCount,
    int candidateCount,
    int beamWidth,
    cudaStream_t stream)
{
    size_t total = size_t(beamCount) * candidateCount;
    // 1. Build segment offsets
    std::vector<int> h_offsets(beamCount+1);
    for (int i = 0; i <= beamCount; i++)
        h_offsets[i] = i * candidateCount;
    int* d_offsets;
    cudaMalloc(&d_offsets, (beamCount+1) * sizeof(int));
    cudaMemcpyAsync(d_offsets, h_offsets.data(), (beamCount+1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 2. Prepare output buffers
    float* d_scores_sorted;
    int*   d_tokens_sorted;
    int*   d_prev_sorted;
    cudaMalloc(&d_scores_sorted, sizeof(float)*total);
    cudaMalloc(&d_tokens_sorted, sizeof(int  )*total);
    cudaMalloc(&d_prev_sorted,   sizeof(int  )*total);

    // 3. Temp storage for CUB
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_bytes,
        scores, d_scores_sorted,
        tokenIds, d_tokens_sorted,
        total, beamCount,
        d_offsets, d_offsets+1,
        0, 32, stream);
    cudaMalloc(&d_temp, temp_bytes);

    // 4. Sort scores/tokenIds
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp, temp_bytes,
        scores, d_scores_sorted,
        tokenIds, d_tokens_sorted,
        total, beamCount,
        d_offsets, d_offsets+1,
        0, 32, stream);

    // 5. Sort scores/prevIndices (keep prevIndices paired)
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_scores_sorted, d_scores_sorted,
        prevIndices, d_prev_sorted,
        total, beamCount,
        d_offsets, d_offsets+1,
        0, 32, stream);

    // 6. Copy top-K to front
    dim3 grid((beamWidth + 255)/256, beamCount);
    copyTopK<<<grid, 256, 0, stream>>>(
        d_scores_sorted, d_tokens_sorted, d_prev_sorted,
        scores, tokenIds, prevIndices,
        beamCount, candidateCount, beamWidth);

    cudaFree(d_offsets);
    cudaFree(d_scores_sorted);
    cudaFree(d_tokens_sorted);
    cudaFree(d_prev_sorted);
    cudaFree(d_temp);
}

} // namespace beam_search
} // namespace whisper 