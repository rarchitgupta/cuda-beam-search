#include "whisper/beam_search/logit_processor.h"
#include "whisper/beam_search/beam_types.h"
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <iostream>

using namespace whisper::beam_search;

int main() {
    // Simple smoke test for ProcessLogits
    const int batch_size = 1;
    const int seq_len = 1;
    const int vocab_size = 128;

    // Prepare dummy logits on host
    std::vector<float> h_logits(batch_size * seq_len * vocab_size, 0.5f);

    // Allocate and copy to device
    float* d_logits = nullptr;
    cudaError_t err = cudaMalloc(&d_logits, h_logits.size() * sizeof(float));
    assert(err == cudaSuccess);
    err = cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // Setup workspace and processor
    BeamSearchWorkspace workspace(1024 * 1024);
    LogitProcessor processor(&workspace, /*temperature=*/1.0f, /*top_k=*/0, /*top_p=*/1.0f);

    // Invoke ProcessLogits (no-op kernel stub) and check it succeeds
    bool success = processor.processLogits(d_logits, batch_size, seq_len, vocab_size);
    assert(success);

    // Clean up
    cudaFree(d_logits);

    std::cout << "ProcessLogits smoke test passed!" << std::endl;
    return 0;
} 