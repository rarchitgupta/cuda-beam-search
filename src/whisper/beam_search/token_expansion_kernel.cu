#include "whisper/beam_search/token_expansion_kernel.h"
#include "whisper/beam_search/cuda_utils.h"

namespace whisper {
namespace beam_search {

// Token expansion: host wrapper launches one thread per beam to do serial top-K scan
void launchTokenExpansion(
    const float* logits,
    int beamCount,
    int vocabSize,
    int topK,
    int* outBeamIds,
    int* outTokenIds,
    float* outScores
) {
    int threads = 1;
    size_t shared_bytes = (size_t)topK * sizeof(float) + (size_t)topK * sizeof(int);
    LAUNCH_AND_CHECK(token_expansion_kernel<<<beamCount, threads, shared_bytes>>>(
        logits, beamCount, vocabSize, topK,
        outBeamIds, outTokenIds, outScores
    ));
}

__global__ void token_expansion_kernel(
    const float* logits,
    int beamCount,
    int vocabSize,
    int topK,
    int* outBeamIds,
    int* outTokenIds,
    float* outScores
) {
    int b = blockIdx.x;
    extern __shared__ float shared_data[];
    float* shared_scores = shared_data;
    int* shared_token_ids = reinterpret_cast<int*>(shared_data + topK);
    if (threadIdx.x == 0) {
        for (int i = 0; i < topK; ++i) {
            shared_scores[i] = -1e30f;
            shared_token_ids[i] = -1;
        }
        for (int t = 0; t < vocabSize; ++t) {
            float score = logits[b * vocabSize + t];
            for (int i = 0; i < topK; ++i) {
                if (score > shared_scores[i]) {
                    for (int j = topK - 1; j > i; --j) {
                        shared_scores[j] = shared_scores[j - 1];
                        shared_token_ids[j] = shared_token_ids[j - 1];
                    }
                    shared_scores[i] = score;
                    shared_token_ids[i] = t;
                    break;
                }
            }
        }
        for (int i = 0; i < topK; ++i) {
            int idx = b * topK + i;
            outBeamIds[idx] = b;
            outTokenIds[idx] = shared_token_ids[i];
            outScores[idx] = shared_scores[i];
        }
    }
}

} // namespace beam_search
} // namespace whisper 