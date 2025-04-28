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
    BeamArray(size_t max_beam_size, BeamSearchWorkspace* workspace);

    ~BeamArray();

    void Reset();

    size_t Size() const { return size_; }

    size_t Capacity() const { return capacity_; }

    int AddToken(const Token& token);

    int AddTokens(const Token* tokens, size_t count);

    void SortByScore();

    // Keep only top beam_width tokens with highest scores
    void Prune(size_t beam_width);

    Token GetToken(size_t index) const;

    void CopyToHost(std::vector<Token>& host_tokens) const;

    float* GetScorePtr() const { return d_scores_; }

    int* GetTokenIdPtr() const { return d_token_ids_; }

    int* GetPrevIndexPtr() const { return d_prev_indices_; }

private:
    // Device memory in SoA layout
    float* d_scores_ = nullptr;
    int* d_token_ids_ = nullptr;
    int* d_prev_indices_ = nullptr;
    int* d_indices_ = nullptr;

    // Host shadow copies for quick access
    std::vector<float> h_scores_;
    std::vector<int> h_token_ids_;
    std::vector<int> h_prev_indices_;

    size_t capacity_ = 0;
    size_t size_ = 0;
    
    BeamSearchWorkspace* workspace_ = nullptr;

    void EnsureCapacity(size_t required_size);
    void AllocateMemory();
    void CopyToHost(std::vector<float>& scores, std::vector<int>& token_ids, std::vector<int>& prev_indices);
};

} // namespace beam_search
} // namespace whisper