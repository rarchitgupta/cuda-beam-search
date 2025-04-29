#include "whisper/beam_search/score_kernel.h"
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <iostream>
#include <cmath>

using namespace whisper::beam_search;

int main() {
    const int beam_count = 3;
    const int vocab_size = 4;

    // Initialize host inputs
    std::vector<float> h_logits = {
        0.1f, 0.2f, 0.3f, 0.4f,
        1.0f, 1.1f, 1.2f, 1.3f,
        2.0f, 2.1f, 2.2f, 2.3f
    };
    std::vector<float> h_prev_scores = {0.5f, 1.5f, 2.5f};

    // Allocate device memory
    float *d_logits, *d_prev, *d_out;
    size_t logits_bytes = beam_count * vocab_size * sizeof(float);
    size_t prev_bytes = beam_count * sizeof(float);
    size_t out_bytes = beam_count * vocab_size * sizeof(float);
    cudaMalloc(&d_logits, logits_bytes);
    cudaMalloc(&d_prev, prev_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_logits, h_logits.data(), logits_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev, h_prev_scores.data(), prev_bytes, cudaMemcpyHostToDevice);

    // Launch scoring
    launchScoreKernel(d_logits, d_prev, beam_count, vocab_size, d_out);

    // Copy back results
    std::vector<float> h_out(beam_count * vocab_size);
    cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);

    // Validate against CPU
    for (int b = 0; b < beam_count; ++b) {
        for (int t = 0; t < vocab_size; ++t) {
            int idx = b * vocab_size + t;
            float expected = h_logits[idx] + h_prev_scores[b];
            assert(std::fabs(h_out[idx] - expected) < 1e-6f);
        }
    }
    std::cout << "Score kernel test passed!" << std::endl;

    cudaFree(d_logits);
    cudaFree(d_prev);
    cudaFree(d_out);
    return 0;
} 