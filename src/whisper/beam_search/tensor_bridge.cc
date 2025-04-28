#include "whisper/beam_search/tensor_bridge.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>  // For std::reverse
#include <stdexcept>
#include "whisper/beam_search/beam_array.h"
#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/logit_processor.h"

namespace whisper {
namespace beam_search {

TensorBridge::TensorBridge(utils::CudaStreamManager* stream_manager) 
    : device_data_(nullptr), batch_size_(0), seq_len_(0), vocab_size_(0), 
      owns_memory_(false), dtype_(TensorDType::FLOAT32),
      stream_manager_(stream_manager), owns_stream_manager_(false),
      beam_workspace_(nullptr), device_tokens_(nullptr), token_count_(0),
      results_copied_(false) {
      
    // Create stream manager if not provided
    if (!stream_manager_) {
        stream_manager_ = new utils::CudaStreamManager();
        owns_stream_manager_ = true;
    }
}

TensorBridge::~TensorBridge() {
    // Free device memory if this object owns it
    if (owns_memory_ && device_data_ != nullptr) {
        try {
            CUDA_CHECK(cudaFree(device_data_), 
                     "Failed to free device memory in TensorBridge");
        } catch (const std::exception& e) {
            // Log error but continue with destruction
            std::cerr << "Error in TensorBridge destructor: " << e.what() << std::endl;
        }
        device_data_ = nullptr;
    }
    
    // Free device tokens if allocated
    if (device_tokens_ != nullptr) {
        try {
            CUDA_CHECK(cudaFree(device_tokens_),
                      "Failed to free device tokens in TensorBridge");
        } catch (const std::exception& e) {
            std::cerr << "Error freeing device tokens: " << e.what() << std::endl;
        }
        device_tokens_ = nullptr;
    }
    
    // Free beam search workspace if allocated
    if (beam_workspace_ != nullptr) {
        try {
            BeamSearchWorkspace* workspace = static_cast<BeamSearchWorkspace*>(beam_workspace_);
            delete workspace;
        } catch (const std::exception& e) {
            std::cerr << "Error freeing beam search workspace: " << e.what() << std::endl;
        }
        beam_workspace_ = nullptr;
    }
    
    // Clean up stream manager if owned
    if (owns_stream_manager_ && stream_manager_) {
        delete stream_manager_;
        stream_manager_ = nullptr;
    }
}

bool TensorBridge::set_logits(float* data, int batch_size, int seq_len, int vocab_size) {
    device_data_ = data;
    batch_size_ = batch_size;
    seq_len_ = seq_len;
    vocab_size_ = vocab_size;
    owns_memory_ = false;  // We don't own this memory - PyTorch does
    dtype_ = TensorDType::FLOAT32;
    
    return true;
}

bool TensorBridge::set_logits_half(void* data, int batch_size, int seq_len, int vocab_size) {
    device_data_ = data;
    batch_size_ = batch_size;
    seq_len_ = seq_len;
    vocab_size_ = vocab_size;
    owns_memory_ = false;  // We don't own this memory - PyTorch does
    dtype_ = TensorDType::FLOAT16;
    
    return true;
}

void* TensorBridge::get_device_data() const {
    return device_data_;
}

TensorDType TensorBridge::get_dtype() const {
    return dtype_;
}

std::vector<int> TensorBridge::get_shape() const {
    return {batch_size_, seq_len_, vocab_size_};
}

utils::CudaStreamManager* TensorBridge::get_stream_manager() const {
    return stream_manager_;
}

bool TensorBridge::execute_beam_search(const BeamSearchConfig& config) {
    if (device_data_ == nullptr) {
        std::cerr << "No logits tensor set for beam search" << std::endl;
        return false;
    }
    
    // Reset result flag
    results_copied_ = false;
    
    // Create workspace if not already created
    if (beam_workspace_ == nullptr) {
        beam_workspace_ = new BeamSearchWorkspace(16 * 1024 * 1024); // 16MB initial size
    }
    
    BeamSearchWorkspace* workspace = static_cast<BeamSearchWorkspace*>(beam_workspace_);
    
    // Create beam array for results - operation only on compute stream
    cudaStream_t compute_stream = stream_manager_->GetComputeStream();
    BeamArray beam_array(config.beam_width, workspace, batch_size_);
    
    // Create logit processor
    LogitProcessor processor(
        workspace,
        config.temperature,
        config.top_k,
        config.top_p,
        stream_manager_
    );
    
    // Process logits based on data type - on compute stream
    bool success = false;
    if (dtype_ == TensorDType::FLOAT32) {
        success = processor.ProcessLogits(
            static_cast<float*>(device_data_),
            batch_size_,
            seq_len_,
            vocab_size_
        );
    } else {
        success = processor.ProcessLogitsHalf(
            device_data_,
            batch_size_,
            seq_len_,
            vocab_size_
        );
    }
    
    if (!success) {
        std::cerr << "Failed to process logits for beam search" << std::endl;
        return false;
    }
    
    // Score and prune tokens - use batched version if enabled, otherwise process first batch only
    if (config.use_batch_processing && batch_size_ > 1) {
        // Process all batches in parallel
        processor.ScoreAndPruneBatched(&beam_array, 0, &beam_array, config.beam_width);
        success = true;
    } else {
        // Process only first batch (legacy mode)
        processor.ScoreAndPrune(&beam_array, 0, 0, &beam_array, config.beam_width);
        success = true;
    }
    
    // Get token count
    token_count_ = beam_array.Size();
    
    // Allocate device memory for tokens if needed
    if (device_tokens_ != nullptr) {
        cudaFree(device_tokens_);
    }
    CUDA_CHECK(cudaMalloc(&device_tokens_, token_count_ * sizeof(Token)),
              "Failed to allocate device memory for tokens");
    
    // Copy tokens to device_tokens_ (still on compute stream)
    success = beam_array.CopyToDevice(static_cast<Token*>(device_tokens_), compute_stream);
    if (!success) {
        std::cerr << "Failed to copy tokens to device memory" << std::endl;
        return false;
    }
    
    // Record event on compute stream to signal completion of token processing
    stream_manager_->RecordComputeEvent();
    
    return true;
}

bool TensorBridge::copy_results_to_host() const {
    if (device_tokens_ == nullptr || token_count_ == 0) {
        return false;
    }
    
    // Wait for compute stream to finish before copying
    // This makes transfer stream wait for compute stream's recorded event
    stream_manager_->WaitForComputeOnTransfer();
    
    // Use transfer stream for device-to-host memory operations
    cudaStream_t transfer_stream = stream_manager_->GetTransferStream();
    
    // Allocate host memory for tokens
    std::vector<Token> tokens(token_count_);
    
    // Copy tokens from device to host - asynchronously on transfer stream
    CUDA_CHECK(cudaMemcpyAsync(
        tokens.data(),
        device_tokens_,
        token_count_ * sizeof(Token),
        cudaMemcpyDeviceToHost,
        transfer_stream
    ), "Failed to copy tokens from device to host");
    
    // Synchronize transfer stream to ensure copy is complete
    stream_manager_->SynchronizeTransfer();
    
    // Process tokens to extract beam search results
    if (batch_size_ > 1) {
        return process_host_tokens_batched(tokens);
    } else {
        return process_host_tokens(tokens);
    }
}

bool TensorBridge::process_host_tokens(const std::vector<Token>& tokens) const {
    if (tokens.empty()) {
        return false;
    }
    
    // Determine beam width from either existing results or default
    const int beam_width = beam_results_.size() > 0 ? beam_results_.size() : 5;
    beam_results_.clear();
    
    for (int i = 0; i < beam_width && i < tokens.size(); i++) {
        std::vector<int> sequence;
        int current_idx = i;
        
        // Follow prev_index links to construct the sequence
        while (current_idx >= 0 && current_idx < tokens.size()) {
            const Token& token = tokens[current_idx];
            sequence.push_back(token.token_id);
            
            if (token.prev_index == current_idx) {
                break;  // Avoid cycles
            }
            
            current_idx = token.prev_index;
        }
        
        // Results are in reverse order, so we need to reverse them
        std::reverse(sequence.begin(), sequence.end());
        beam_results_.push_back(sequence);
    }
    
    results_copied_ = true;
    return true;
}

bool TensorBridge::process_host_tokens_batched(const std::vector<Token>& tokens) const {
    if (tokens.empty()) {
        return false;
    }
    
    // Clear batch results
    batch_beam_results_.clear();
    batch_beam_results_.resize(batch_size_);
    
    // Determine beam width from previous results or default
    const int beam_width = beam_results_.size() > 0 ? beam_results_.size() : 5;
    
    // Group tokens by batch index
    std::vector<std::vector<int>> batch_tokens(batch_size_);
    
    for (int i = 0; i < tokens.size(); i++) {
        const Token& token = tokens[i];
        int batch_idx = token.batch_index;
        
        // Skip invalid batch indices
        if (batch_idx < 0 || batch_idx >= batch_size_) {
            continue;
        }
        
        // Store token index for this batch
        batch_tokens[batch_idx].push_back(i);
    }
    
    // Process each batch separately
    for (int batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
        const auto& token_indices = batch_tokens[batch_idx];
        
        // Skip empty batches
        if (token_indices.empty()) {
            continue;
        }
        
        // Get beam results for this batch
        std::vector<std::vector<int>> batch_results;
        
        // Process up to beam_width tokens
        for (int i = 0; i < beam_width && i < token_indices.size(); i++) {
            std::vector<int> sequence;
            int current_idx = token_indices[i];
            
            // Follow prev_index links to construct the sequence
            while (current_idx >= 0 && current_idx < tokens.size()) {
                const Token& token = tokens[current_idx];
                sequence.push_back(token.token_id);
                
                if (token.prev_index == current_idx) {
                    break;  // Avoid cycles
                }
                
                current_idx = token.prev_index;
            }
            
            // Results are in reverse order, so we need to reverse them
            std::reverse(sequence.begin(), sequence.end());
            batch_results.push_back(sequence);
        }
        
        // Store results for this batch
        batch_beam_results_[batch_idx] = batch_results;
    }
    
    // Also update the single-batch results to maintain backward compatibility
    // Use first batch's results
    if (!batch_beam_results_.empty() && !batch_beam_results_[0].empty()) {
        beam_results_ = batch_beam_results_[0];
    } else {
        beam_results_.clear();
    }
    
    results_copied_ = true;
    return true;
}

std::vector<std::vector<int>> TensorBridge::get_beam_search_results() const {
    // If results haven't been copied yet, copy them
    if (!results_copied_) {
        copy_results_to_host();
    }
    
    return beam_results_;
}

std::vector<std::vector<std::vector<int>>> TensorBridge::get_batch_beam_search_results() const {
    // If results haven't been copied yet, copy them
    if (!results_copied_) {
        copy_results_to_host();
    }
    
    return batch_beam_results_;
}

void TensorBridge::reset_beam_search() {
    beam_results_.clear();
    batch_beam_results_.clear();
    results_copied_ = false;
    
    // Free device tokens if allocated
    if (device_tokens_ != nullptr) {
        cudaFree(device_tokens_);
        device_tokens_ = nullptr;
    }
    
    token_count_ = 0;
    
    // Reset workspace if it exists
    if (beam_workspace_ != nullptr) {
        BeamSearchWorkspace* workspace = static_cast<BeamSearchWorkspace*>(beam_workspace_);
        workspace->Reset();
    }
}

} // namespace beam_search
} // namespace whisper 