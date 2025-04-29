#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace whisper {
namespace beam_search {

// GPU kernel to expand each beam to its top-K candidate tokens.
__global__ void token_expansion_kernel(
    const float* logits,
    int beamCount,
    int vocabSize,
    int topK,
    int* outBeamIds,
    int* outTokenIds,
    float* outScores
);

// Launches the token_expansion_kernel on the device.
void launchTokenExpansion(
    const float* logits,
    int beamCount,
    int vocabSize,
    int topK,
    int* outBeamIds,
    int* outTokenIds,
    float* outScores
);

}  // namespace beam_search
}  // namespace whisper 