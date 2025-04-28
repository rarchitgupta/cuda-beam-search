#pragma once

#include "whisper/beam_search/beam_types.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace whisper {
namespace beam_search {

// Manages beam tokens using Structure of Arrays (SoA) pattern for efficient GPU memory access
class BeamArray {
public:
    BeamArray(size_t max_beam_size, BeamSearchWorkspace* workspace, size_t max_batch_size = 1);

    ~BeamArray();

    void Reset();

    size_t Size() const { return size_; }

    size_t Capacity() const { return capacity_; }
    
    size_t BatchSize() const { return max_batch_size_; }

    int AddToken(const Token& token);

    int AddTokens(const Token* tokens, size_t count);
    
    // Get token count for a specific batch
    size_t GetBatchTokenCount(int batch_idx) const;

    void SortByScore();
    
    // Sort tokens within each batch separately
    void SortByScoreBatched();

    // Keep only top beam_width tokens with highest scores
    void Prune(size_t beam_width);
    
    // Prune each batch separately to keep only top beam_width tokens
    void PruneBatched(size_t beam_width);

    Token GetToken(size_t index) const;

    // Copy to host - blocking
    void CopyToHost(std::vector<Token>& host_tokens) const;
    
    // Copy to host using specified stream - non-blocking
    void CopyToHostAsync(std::vector<Token>& host_tokens, cudaStream_t stream) const;
    
    // Copy to device (Token array)
    bool CopyToDevice(Token* device_tokens, cudaStream_t stream) const;

    float* GetScorePtr() const { return d_scores_; }

    int* GetTokenIdPtr() const { return d_token_ids_; }

    int* GetPrevIndexPtr() const { return d_prev_indices_; }
    
    int* GetBatchIndexPtr() const { return d_batch_indices_; }

private:
    // Device memory in SoA layout
    float* d_scores_ = nullptr;
    int* d_token_ids_ = nullptr;
    int* d_prev_indices_ = nullptr;
    int* d_batch_indices_ = nullptr;
    int* d_indices_ = nullptr;

    // Host shadow copies for quick access
    std::vector<float> h_scores_;
    std::vector<int> h_token_ids_;
    std::vector<int> h_prev_indices_;
    std::vector<int> h_batch_indices_;
    
    // Batch tracking
    size_t max_batch_size_ = 1;
    std::vector<size_t> batch_token_counts_;

    size_t capacity_ = 0;
    size_t size_ = 0;
    
    BeamSearchWorkspace* workspace_ = nullptr;

    void EnsureCapacity(size_t required_size);
    void AllocateMemory();
    void CopyToHost(std::vector<float>& scores, std::vector<int>& token_ids, std::vector<int>& prev_indices, std::vector<int>& batch_indices);
    void CopyToHostAsync(std::vector<float>& scores, std::vector<int>& token_ids, std::vector<int>& prev_indices, std::vector<int>& batch_indices, cudaStream_t stream);
};

} // namespace beam_search
} // namespace whisper