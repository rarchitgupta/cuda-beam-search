#include "whisper/beam_search/score_kernel.h"
#include "whisper/beam_search/cuda_utils.h"

namespace whisper {
namespace beam_search {

__global__ void score_kernel(
    const float* logits,
    const float* prevScores,
    int beamCount,
    int vocabSize,
    float* outScores
) {
    int b = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < beamCount && t < vocabSize) {
        int idx = b * vocabSize + t;
        outScores[idx] = logits[idx] + prevScores[b];
    }
}

void launchScoreKernel(
    const float* logits,
    const float* prevScores,
    int beamCount,
    int vocabSize,
    float* outScores
) {
    const int blockSize = 256;
    dim3 blockDim(blockSize);
    dim3 gridDim((vocabSize + blockSize - 1) / blockSize, beamCount);
    LAUNCH_AND_CHECK(score_kernel<<<gridDim, blockDim>>>(
        logits, prevScores,
        beamCount, vocabSize,
        outScores
    ));
}

} // namespace beam_search
} // namespace whisper 