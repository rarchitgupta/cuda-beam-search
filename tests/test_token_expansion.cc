#include "whisper/beam_search/token_expansion_kernel.h"
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>  // for std::fabs

using namespace whisper::beam_search;

int main() {
    const int beam_count = 2;
    const int vocab_size = 5;
    const int K = 3;

    // Create deterministic logits for two beams
    // beam 0: [0.1, 0.5, 0.2, 0.9, 0.3]
    // beam 1: [1.0, 0.4, 0.6, 0.8, 0.7]
    std::vector<float> h_logits = {
        0.1f, 0.5f, 0.2f, 0.9f, 0.3f,
        1.0f, 0.4f, 0.6f, 0.8f, 0.7f
    };

    // Allocate device memory
    float* d_logits = nullptr;
    int* d_beam_ids = nullptr;
    int* d_token_ids = nullptr;
    float* d_scores = nullptr;
    size_t logits_bytes = beam_count * vocab_size * sizeof(float);
    size_t out_bytes = beam_count * K * sizeof(int);
    size_t scores_bytes = beam_count * K * sizeof(float);

    cudaMalloc(&d_logits, logits_bytes);
    cudaMalloc(&d_beam_ids, out_bytes);
    cudaMalloc(&d_token_ids, out_bytes);
    cudaMalloc(&d_scores, scores_bytes);

    cudaMemcpy(d_logits, h_logits.data(), logits_bytes, cudaMemcpyHostToDevice);

    // Launch expansion kernel
    launchTokenExpansion(
        d_logits, beam_count, vocab_size, K,
        d_beam_ids, d_token_ids, d_scores
    );

    // Copy results back
    std::vector<int> h_beam_ids(beam_count * K);
    std::vector<int> h_token_ids(beam_count * K);
    std::vector<float> h_scores(beam_count * K);
    cudaMemcpy(h_beam_ids.data(), d_beam_ids, out_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_token_ids.data(), d_token_ids, out_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scores.data(), d_scores, scores_bytes, cudaMemcpyDeviceToHost);

    // Compute expected on host
    std::vector<int> expected_beam_ids;
    std::vector<int> expected_token_ids;
    std::vector<float> expected_scores;
    expected_beam_ids.reserve(beam_count * K);
    expected_token_ids.reserve(beam_count * K);
    expected_scores.reserve(beam_count * K);

    for (int b = 0; b < beam_count; ++b) {
        std::vector<std::pair<float,int>> pairs;
        for (int t = 0; t < vocab_size; ++t) {
            pairs.emplace_back(h_logits[b * vocab_size + t], t);
        }
        // sort descending by score
        std::sort(pairs.begin(), pairs.end(), [](auto &a, auto &b) {
            return a.first > b.first;
        });
        for (int i = 0; i < K; ++i) {
            expected_beam_ids.push_back(b);
            expected_token_ids.push_back(pairs[i].second);
            expected_scores.push_back(pairs[i].first);
        }
    }

    // Validate
    for (int idx = 0; idx < beam_count * K; ++idx) {
        assert(h_beam_ids[idx] == expected_beam_ids[idx]);
        assert(h_token_ids[idx] == expected_token_ids[idx]);
        assert(std::fabs(h_scores[idx] - expected_scores[idx]) < 1e-6f);
    }

    std::cout << "Token expansion test passed!" << std::endl;

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_beam_ids);
    cudaFree(d_token_ids);
    cudaFree(d_scores);

    return 0;
} 