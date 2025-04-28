#pragma once

#include <cstdint>
#include <vector>
#include "whisper/utils/cuda_stream_manager.h"
#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_config.h"

namespace whisper {
namespace beam_search {

enum class TensorDType {
    FLOAT32,
    FLOAT16
};

// Interface for passing PyTorch/TensorFlow tensors to CUDA beam search
class TensorBridge {
public:
    TensorBridge(utils::CudaStreamManager* stream_manager = nullptr);
    ~TensorBridge();

    // Associate logits tensor from ML framework (doesn't take ownership)
    bool set_logits(float* data, int batch_size, int seq_len, int vocab_size);
    
    // Half-precision version
    bool set_logits_half(void* data, int batch_size, int seq_len, int vocab_size);

    void* get_device_data() const;
    TensorDType get_dtype() const;

    std::vector<int> get_shape() const;
    
    // Get stream manager or nullptr if not used
    utils::CudaStreamManager* get_stream_manager() const;

    // Beam search methods
    bool execute_beam_search(const BeamSearchConfig& config);
    
    // Overloaded beam search method with explicit parameters for C API compatibility
    bool execute_beam_search(const void* logits, int batch_size, int seq_len, 
                           int vocab_size, float temperature, int top_k, 
                           float top_p, int beam_width, int* tokens_out);
    
    // Get beam search results for single batch
    std::vector<std::vector<int>> get_beam_search_results() const;
    
    // Get beam search results for all batches
    std::vector<std::vector<std::vector<int>>> get_batch_beam_search_results() const;
    
    void reset_beam_search();

private:
    // Asynchronously copy results from device to host
    bool copy_results_to_host() const;
    
    // Process tokens on host to extract beam search results
    bool process_host_tokens(const std::vector<Token>& tokens) const;
    
    // Process tokens on host to extract batched beam search results
    bool process_host_tokens_batched(const std::vector<Token>& tokens) const;
    
    void* device_data_;
    int batch_size_;
    int seq_len_;
    int vocab_size_;
    bool owns_memory_;
    TensorDType dtype_;
    utils::CudaStreamManager* stream_manager_;
    bool owns_stream_manager_;
    
    // Beam search state
    void* beam_workspace_;
    void* device_tokens_;
    int token_count_;
    mutable std::vector<std::vector<int>> beam_results_;
    mutable std::vector<std::vector<std::vector<int>>> batch_beam_results_;
    mutable bool results_copied_;
};

} // namespace beam_search
} // namespace whisper