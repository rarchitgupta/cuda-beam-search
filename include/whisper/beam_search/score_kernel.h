#pragma once
#include <cuda_runtime.h>

namespace whisper {
namespace beam_search {

// GPU kernel to score each token by adding previous beam scores.
__global__ void score_kernel(
    const float* logits,
    const float* prevScores,
    int beamCount,
    int vocabSize,
    float* outScores
);

// Launches the score kernel on the device.
void launchScoreKernel(
    const float* logits,
    const float* prevScores,
    int beamCount,
    int vocabSize,
    float* outScores
);

} // namespace beam_search
} // namespace whisper 