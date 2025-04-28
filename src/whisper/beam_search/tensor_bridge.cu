#include "whisper/beam_search/tensor_bridge.h"
#include <cuda_runtime.h>
#include <iostream>

namespace whisper {
namespace beam_search {

// Helper function to check if CUDA context is valid
namespace {
bool CheckCudaContextValid() {
    int* d_check;
    cudaError_t err = cudaMalloc(&d_check, sizeof(int));
    if (err != cudaSuccess) {
        return false;
    }
    
    err = cudaMemset(d_check, 0, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_check);
        return false;
    }
    
    err = cudaFree(d_check);
    if (err != cudaSuccess) {
        return false;
    }
    
    return true;
}
}  // namespace

bool TensorBridge::execute_beam_search(const void* logits, int batch_size, int seq_len, 
                                    int vocab_size, float temperature, int top_k, 
                                    float top_p, int beam_width, int* tokens_out) {
    // Use batched processing if supported for this batch size
    bool use_batched = batch_size > 1;
    
    // Verify CUDA context is valid
    if (!CheckCudaContextValid()) {
        std::cerr << "CUDA context is not valid for beam search execution" << std::endl;
        return false;
    }
    
    // Create configuration
    BeamSearchConfig config;
    config.beam_width = beam_width;
    config.temperature = temperature;
    config.top_k = top_k;
    config.top_p = top_p;
    config.use_batch_processing = use_batched;
    
    // Set logits data based on the fact that this variant of the function 
    // always expects float32 logits
    bool success = set_logits(static_cast<float*>(const_cast<void*>(logits)), 
                             batch_size, seq_len, vocab_size);
    if (!success) {
        std::cerr << "Failed to set logits data" << std::endl;
        return false;
    }
    
    // Execute beam search
    success = execute_beam_search(config);
    if (!success) {
        return false;
    }
    
    // Copy results to host
    success = copy_results_to_host();
    if (!success) {
        std::cerr << "Failed to copy results to host" << std::endl;
        return false;
    }
    
    // Get best result for each batch
    auto results = get_batch_beam_search_results();
    
    // Extract tokens from the best beam of each batch
    int token_offset = 0;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        if (batch_idx < static_cast<int>(results.size()) && 
            !results[batch_idx].empty() && 
            !results[batch_idx][0].empty()) {
            
            const auto& best_sequence = results[batch_idx][0];
            int seq_len = best_sequence.size();
            
            // Copy tokens to output
            for (int i = 0; i < seq_len; i++) {
                tokens_out[token_offset + i] = best_sequence[i];
            }
            
            token_offset += seq_len;
        }
    }
    
    return true;
}

} // namespace beam_search
} // namespace whisper 