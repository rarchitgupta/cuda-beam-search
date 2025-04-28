#include "whisper/beam_search/beam_array.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <stdexcept>

namespace whisper {
namespace beam_search {

// CUDA kernel for generating indices for sorting
__global__ void GenerateIndicesKernel(int* indices, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        indices[idx] = idx;
    }
}

// CUDA kernel for batched token addition
__global__ void AddTokensKernel(float* d_scores, int* d_token_ids, int* d_prev_indices, 
                               const float* new_scores, const int* new_token_ids, const int* new_prev_indices,
                               size_t start_idx, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_scores[start_idx + idx] = new_scores[idx];
        d_token_ids[start_idx + idx] = new_token_ids[idx];
        d_prev_indices[start_idx + idx] = new_prev_indices[idx];
    }
}

// CUDA kernel for reordering data based on sorted indices
__global__ void ReorderDataKernel(float* d_sorted_scores, int* d_sorted_token_ids, int* d_sorted_prev_indices,
                                 const float* d_scores, const int* d_token_ids, const int* d_prev_indices,
                                 const int* d_indices, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int source_idx = d_indices[idx];
        d_sorted_scores[idx] = d_scores[source_idx];
        d_sorted_token_ids[idx] = d_token_ids[source_idx];
        d_sorted_prev_indices[idx] = d_prev_indices[source_idx];
    }
}

// CUDA comparator for token sorting (descending order)
struct TokenScoreComparator {
    const float* scores;
    
    TokenScoreComparator(const float* s) : scores(s) {}
    
    __host__ __device__ bool operator()(int a, int b) const {
        return scores[a] > scores[b];
    }
};

constexpr int kBlockSize = 256;
constexpr size_t kAlignment = 128; // Align to 128 bytes for optimal memory access

BeamArray::BeamArray(size_t max_beam_size, BeamSearchWorkspace* workspace)
    : capacity_(max_beam_size), workspace_(workspace) {
    // Round up capacity to alignment boundary
    capacity_ = (capacity_ + kAlignment - 1) / kAlignment * kAlignment;
    AllocateMemory();
    Reset();
}

BeamArray::~BeamArray() {
    // Memory managed by workspace
}

void BeamArray::Reset() {
    size_ = 0;
    
    h_scores_.clear();
    h_token_ids_.clear();
    h_prev_indices_.clear();
}

void BeamArray::AllocateMemory() {
    size_t token_size = capacity_ * sizeof(float);
    size_t index_size = capacity_ * sizeof(int);
    
    d_scores_ = static_cast<float*>(workspace_->Allocate(token_size, kAlignment));
    d_token_ids_ = static_cast<int*>(workspace_->Allocate(index_size, kAlignment));
    d_prev_indices_ = static_cast<int*>(workspace_->Allocate(index_size, kAlignment));
    d_indices_ = static_cast<int*>(workspace_->Allocate(index_size, kAlignment));
    
    if (!d_scores_ || !d_token_ids_ || !d_prev_indices_ || !d_indices_) {
        throw std::runtime_error("Failed to allocate device memory for BeamArray");
    }
    
    h_scores_.reserve(capacity_);
    h_token_ids_.reserve(capacity_);
    h_prev_indices_.reserve(capacity_);
}

void BeamArray::EnsureCapacity(size_t required_size) {
    if (required_size <= capacity_) {
        return;
    }
    
    size_t new_capacity = std::max(required_size, capacity_ * 2);
    new_capacity = (new_capacity + kAlignment - 1) / kAlignment * kAlignment;
    
    float* old_scores = d_scores_;
    int* old_token_ids = d_token_ids_;
    int* old_prev_indices = d_prev_indices_;
    
    size_t new_token_size = new_capacity * sizeof(float);
    size_t new_index_size = new_capacity * sizeof(int);
    
    d_scores_ = static_cast<float*>(workspace_->Allocate(new_token_size, kAlignment));
    d_token_ids_ = static_cast<int*>(workspace_->Allocate(new_index_size, kAlignment));
    d_prev_indices_ = static_cast<int*>(workspace_->Allocate(new_index_size, kAlignment));
    d_indices_ = static_cast<int*>(workspace_->Allocate(new_index_size, kAlignment));
    
    if (!d_scores_ || !d_token_ids_ || !d_prev_indices_ || !d_indices_) {
        throw std::runtime_error("Failed to reallocate device memory for BeamArray");
    }
    
    // Copy old data to new memory
    if (size_ > 0) {
        cudaMemcpy(d_scores_, old_scores, size_ * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_token_ids_, old_token_ids, size_ * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_prev_indices_, old_prev_indices, size_ * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    
    capacity_ = new_capacity;
    
    h_scores_.reserve(capacity_);
    h_token_ids_.reserve(capacity_);
    h_prev_indices_.reserve(capacity_);
}

int BeamArray::AddToken(const Token& token) {
    if (size_ >= capacity_) {
        // Try to increase capacity
        EnsureCapacity(size_ + 1);
        if (size_ >= capacity_) {
            return -1;  // Failed to increase capacity
        }
    }
    
    // Copy token to device memory
    cudaMemcpy(d_scores_ + size_, &token.score, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_token_ids_ + size_, &token.token_id, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_indices_ + size_, &token.prev_index, sizeof(int), cudaMemcpyHostToDevice);
    
    // Add to host shadow copies for quick access
    h_scores_.push_back(token.score);
    h_token_ids_.push_back(token.token_id);
    h_prev_indices_.push_back(token.prev_index);
    
    return size_++;
}

int BeamArray::AddTokens(const Token* tokens, size_t count) {
    if (count == 0) {
        return 0;
    }
    
    if (size_ + count > capacity_) {
        EnsureCapacity(size_ + count);
        if (size_ + count > capacity_) {
            count = capacity_ - size_;  // Truncate to available space
            if (count == 0) {
                return 0;  // No space available
            }
        }
    }
    
    std::vector<float> new_scores(count);
    std::vector<int> new_token_ids(count);
    std::vector<int> new_prev_indices(count);
    
    for (size_t i = 0; i < count; i++) {
        new_scores[i] = tokens[i].score;
        new_token_ids[i] = tokens[i].token_id;
        new_prev_indices[i] = tokens[i].prev_index;
        
        h_scores_.push_back(tokens[i].score);
        h_token_ids_.push_back(tokens[i].token_id);
        h_prev_indices_.push_back(tokens[i].prev_index);
    }
    
    float* d_new_scores;
    int* d_new_token_ids;
    int* d_new_prev_indices;
    
    cudaMalloc(&d_new_scores, count * sizeof(float));
    cudaMalloc(&d_new_token_ids, count * sizeof(int));
    cudaMalloc(&d_new_prev_indices, count * sizeof(int));
    
    cudaMemcpy(d_new_scores, new_scores.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_token_ids, new_token_ids.data(), count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_prev_indices, new_prev_indices.data(), count * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel to add tokens in parallel
    int blocks = (count + kBlockSize - 1) / kBlockSize;
    AddTokensKernel<<<blocks, kBlockSize>>>(d_scores_, d_token_ids_, d_prev_indices_,
                                           d_new_scores, d_new_token_ids, d_new_prev_indices,
                                           size_, count);
    
    cudaFree(d_new_scores);
    cudaFree(d_new_token_ids);
    cudaFree(d_new_prev_indices);
    
    size_ += count;
    return count;
}

void BeamArray::SortByScore() {
    if (size_ <= 1) {
        return;  // Nothing to sort
    }
    
    // Generate indices for sorting
    int blocks = (size_ + kBlockSize - 1) / kBlockSize;
    GenerateIndicesKernel<<<blocks, kBlockSize>>>(d_indices_, size_);
    
    // Sort indices based on scores (descending order)
    thrust::device_ptr<int> d_indices_ptr(d_indices_);
    thrust::sort(thrust::device, d_indices_ptr, d_indices_ptr + size_, 
                TokenScoreComparator(d_scores_));
    
    // Allocate temporary memory for sorted data
    float* d_sorted_scores;
    int* d_sorted_token_ids;
    int* d_sorted_prev_indices;
    
    cudaMalloc(&d_sorted_scores, size_ * sizeof(float));
    cudaMalloc(&d_sorted_token_ids, size_ * sizeof(int));
    cudaMalloc(&d_sorted_prev_indices, size_ * sizeof(int));
    
    // Reorder data in parallel based on sorted indices
    ReorderDataKernel<<<blocks, kBlockSize>>>(d_sorted_scores, d_sorted_token_ids, d_sorted_prev_indices,
                                             d_scores_, d_token_ids_, d_prev_indices_,
                                             d_indices_, size_);
    
    // Copy sorted data back
    cudaMemcpy(d_scores_, d_sorted_scores, size_ * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_token_ids_, d_sorted_token_ids, size_ * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_prev_indices_, d_sorted_prev_indices, size_ * sizeof(int), cudaMemcpyDeviceToDevice);
    
    cudaFree(d_sorted_scores);
    cudaFree(d_sorted_token_ids);
    cudaFree(d_sorted_prev_indices);
    
    // Update host shadow copies
    CopyToHost(h_scores_, h_token_ids_, h_prev_indices_);
}

void BeamArray::Prune(size_t beam_width) {
    if (size_ <= beam_width) {
        return;  // No pruning needed
    }
    
    // Sort first to ensure we keep the best tokens
    SortByScore();
    
    // Truncate to beam width
    size_ = beam_width;
    
    // Truncate host shadow copies
    h_scores_.resize(size_);
    h_token_ids_.resize(size_);
    h_prev_indices_.resize(size_);
}

Token BeamArray::GetToken(size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("Token index out of range");
    }
    
    // Use host shadow copies for fast access if available
    if (index < h_scores_.size()) {
        return Token(h_scores_[index], h_token_ids_[index], h_prev_indices_[index]);
    }
    
    // Otherwise, fetch from device
    Token token;
    cudaMemcpy(&token.score, d_scores_ + index, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&token.token_id, d_token_ids_ + index, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&token.prev_index, d_prev_indices_ + index, sizeof(int), cudaMemcpyDeviceToHost);
    
    return token;
}

void BeamArray::CopyToHost(std::vector<Token>& host_tokens) const {
    host_tokens.resize(size_);
    
    if (size_ == 0) {
        return;
    }
    
    std::vector<float> scores(size_);
    std::vector<int> token_ids(size_);
    std::vector<int> prev_indices(size_);
    
    // Copy from device to host
    cudaMemcpy(scores.data(), d_scores_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(token_ids.data(), d_token_ids_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(prev_indices.data(), d_prev_indices_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Construct tokens
    for (size_t i = 0; i < size_; i++) {
        host_tokens[i] = Token(scores[i], token_ids[i], prev_indices[i]);
    }
}

void BeamArray::CopyToHost(std::vector<float>& scores, std::vector<int>& token_ids, std::vector<int>& prev_indices) {
    scores.resize(size_);
    token_ids.resize(size_);
    prev_indices.resize(size_);
    
    if (size_ == 0) {
        return;
    }
    
    cudaMemcpy(scores.data(), d_scores_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(token_ids.data(), d_token_ids_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(prev_indices.data(), d_prev_indices_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
}

} // namespace beam_search
} // namespace whisper 