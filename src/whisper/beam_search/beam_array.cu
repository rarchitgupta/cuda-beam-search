#include "whisper/beam_search/beam_array.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
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
__global__ void AddTokensKernel(float* d_scores, int* d_token_ids, int* d_prev_indices, int* d_batch_indices,
                               const float* new_scores, const int* new_token_ids, const int* new_prev_indices, const int* new_batch_indices,
                               size_t start_idx, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        d_scores[start_idx + idx] = new_scores[idx];
        d_token_ids[start_idx + idx] = new_token_ids[idx];
        d_prev_indices[start_idx + idx] = new_prev_indices[idx];
        d_batch_indices[start_idx + idx] = new_batch_indices[idx];
    }
}

// CUDA kernel for reordering data based on sorted indices
__global__ void ReorderDataKernel(float* d_sorted_scores, int* d_sorted_token_ids, int* d_sorted_prev_indices, int* d_sorted_batch_indices,
                                 const float* d_scores, const int* d_token_ids, const int* d_prev_indices, const int* d_batch_indices,
                                 const int* d_indices, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int source_idx = d_indices[idx];
        d_sorted_scores[idx] = d_scores[source_idx];
        d_sorted_token_ids[idx] = d_token_ids[source_idx];
        d_sorted_prev_indices[idx] = d_prev_indices[source_idx];
        d_sorted_batch_indices[idx] = d_batch_indices[source_idx];
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

// CUDA comparator for batch-aware token sorting (descending order within each batch)
struct BatchedTokenScoreComparator {
    const float* scores;
    const int* batch_indices;
    
    BatchedTokenScoreComparator(const float* s, const int* b) : scores(s), batch_indices(b) {}
    
    __host__ __device__ bool operator()(int a, int b) const {
        if (batch_indices[a] != batch_indices[b]) {
            return batch_indices[a] < batch_indices[b]; // Sort by batch first
        }
        return scores[a] > scores[b]; // Then by score (descending)
    }
};

// CUDA kernel to convert from SoA to AoS format
__global__ void convertToAoSKernel(float* scores, int* token_ids, int* prev_indices, int* batch_indices,
                                  Token* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx].score = scores[idx];
        output[idx].token_id = token_ids[idx];
        output[idx].prev_index = prev_indices[idx];
        output[idx].batch_index = batch_indices[idx];
    }
}

// CUDA kernel to convert from AoS to SoA format
__global__ void convertToSoAKernel(const Token* input, float* scores, int* token_ids, 
                                  int* prev_indices, int* batch_indices, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scores[idx] = input[idx].score;
        token_ids[idx] = input[idx].token_id;
        prev_indices[idx] = input[idx].prev_index;
        batch_indices[idx] = input[idx].batch_index;
    }
}

BeamArray::BeamArray(size_t max_beam_size, BeamSearchWorkspace* workspace, size_t max_batch_size)
    : capacity_(max_beam_size), size_(0), workspace_(workspace), max_batch_size_(max_batch_size) {
    
    // Initialize batch token counts
    batch_token_counts_.resize(max_batch_size_, 0);
    
    // Allocate device memory
    AllocateMemory();
}

BeamArray::~BeamArray() {
    // No need to free memory - workspace will handle it
}

void BeamArray::Reset() {
    size_ = 0;
    // Reset batch token counts
    std::fill(batch_token_counts_.begin(), batch_token_counts_.end(), 0);
}

size_t BeamArray::GetBatchTokenCount(int batch_idx) const {
    if (batch_idx < 0 || batch_idx >= static_cast<int>(max_batch_size_)) {
        return 0;
    }
    return batch_token_counts_[batch_idx];
}

void BeamArray::AllocateMemory() {
    size_t bytes_float = capacity_ * sizeof(float);
    size_t bytes_int = capacity_ * sizeof(int);
    
    try {
        d_scores_ = static_cast<float*>(workspace_->Allocate(bytes_float));
        d_token_ids_ = static_cast<int*>(workspace_->Allocate(bytes_int));
        d_prev_indices_ = static_cast<int*>(workspace_->Allocate(bytes_int));
        d_batch_indices_ = static_cast<int*>(workspace_->Allocate(bytes_int));
        d_indices_ = static_cast<int*>(workspace_->Allocate(bytes_int));
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw std::runtime_error("Failed to allocate device memory for BeamArray");
    }
    
    // Reserve host vectors
    h_scores_.reserve(capacity_);
    h_token_ids_.reserve(capacity_);
    h_prev_indices_.reserve(capacity_);
    h_batch_indices_.reserve(capacity_);
}

void BeamArray::EnsureCapacity(size_t required_size) {
    if (required_size <= capacity_) {
        return;
    }
    
    // Double capacity until it's sufficient
    size_t new_capacity = capacity_;
    while (new_capacity < required_size) {
        new_capacity *= 2;
    }
    
    // Allocate new arrays
    size_t bytes_float = new_capacity * sizeof(float);
    size_t bytes_int = new_capacity * sizeof(int);
    
    try {
        float* new_scores = static_cast<float*>(workspace_->Allocate(bytes_float));
        int* new_token_ids = static_cast<int*>(workspace_->Allocate(bytes_int));
        int* new_prev_indices = static_cast<int*>(workspace_->Allocate(bytes_int));
        int* new_batch_indices = static_cast<int*>(workspace_->Allocate(bytes_int));
        int* new_indices = static_cast<int*>(workspace_->Allocate(bytes_int));
        
        // Copy existing data
        cudaMemcpy(new_scores, d_scores_, size_ * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_token_ids, d_token_ids_, size_ * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_prev_indices, d_prev_indices_, size_ * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_batch_indices, d_batch_indices_, size_ * sizeof(int), cudaMemcpyDeviceToDevice);
        
        d_scores_ = new_scores;
        d_token_ids_ = new_token_ids;
        d_prev_indices_ = new_prev_indices;
        d_batch_indices_ = new_batch_indices;
        d_indices_ = new_indices;
        
        capacity_ = new_capacity;
        
        // Reserve host vectors
        h_scores_.reserve(capacity_);
        h_token_ids_.reserve(capacity_);
        h_prev_indices_.reserve(capacity_);
        h_batch_indices_.reserve(capacity_);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw std::runtime_error("Failed to reallocate device memory for BeamArray");
    }
}

int BeamArray::AddToken(const Token& token) {
    EnsureCapacity(size_ + 1);
    
    // Copy token data to device
    cudaMemcpy(d_scores_ + size_, &token.score, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_token_ids_ + size_, &token.token_id, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_indices_ + size_, &token.prev_index, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch_indices_ + size_, &token.batch_index, sizeof(int), cudaMemcpyHostToDevice);
    
    // Update host shadow copies
    if (h_scores_.size() == size_) {
        h_scores_.push_back(token.score);
        h_token_ids_.push_back(token.token_id);
        h_prev_indices_.push_back(token.prev_index);
        h_batch_indices_.push_back(token.batch_index);
    } else {
        if (size_ < h_scores_.size()) {
            h_scores_[size_] = token.score;
            h_token_ids_[size_] = token.token_id;
            h_prev_indices_[size_] = token.prev_index;
            h_batch_indices_[size_] = token.batch_index;
        } else {
            h_scores_.resize(size_ + 1);
            h_token_ids_.resize(size_ + 1);
            h_prev_indices_.resize(size_ + 1);
            h_batch_indices_.resize(size_ + 1);
            h_scores_[size_] = token.score;
            h_token_ids_[size_] = token.token_id;
            h_prev_indices_[size_] = token.prev_index;
            h_batch_indices_[size_] = token.batch_index;
        }
    }
    
    // Update batch token count
    if (token.batch_index >= 0 && token.batch_index < static_cast<int>(max_batch_size_)) {
        batch_token_counts_[token.batch_index]++;
    }
    
    // Return index of added token
    return size_++;
}

int BeamArray::AddTokens(const Token* tokens, size_t count) {
    if (count == 0) {
        return -1;
    }
    
    EnsureCapacity(size_ + count);
    
    // Copy tokens to device
    for (size_t i = 0; i < count; i++) {
        const Token& token = tokens[i];
        
        // Copy token data to device
        cudaMemcpy(d_scores_ + size_ + i, &token.score, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_token_ids_ + size_ + i, &token.token_id, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prev_indices_ + size_ + i, &token.prev_index, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_batch_indices_ + size_ + i, &token.batch_index, sizeof(int), cudaMemcpyHostToDevice);
        
        // Update host shadow copies
        if (size_ + i < h_scores_.size()) {
            h_scores_[size_ + i] = token.score;
            h_token_ids_[size_ + i] = token.token_id;
            h_prev_indices_[size_ + i] = token.prev_index;
            h_batch_indices_[size_ + i] = token.batch_index;
        } else {
            h_scores_.push_back(token.score);
            h_token_ids_.push_back(token.token_id);
            h_prev_indices_.push_back(token.prev_index);
            h_batch_indices_.push_back(token.batch_index);
        }
        
        // Update batch token count
        if (token.batch_index >= 0 && token.batch_index < static_cast<int>(max_batch_size_)) {
            batch_token_counts_[token.batch_index]++;
        }
    }
    
    // Update size
    size_ += count;
    
    return count;
}

void BeamArray::SortByScore() {
    if (size_ <= 1) {
        return;
    }
    
    // Initialize the indices
    thrust::device_ptr<int> indices_ptr(d_indices_);
    thrust::sequence(thrust::device, indices_ptr, indices_ptr + size_);
    
    // Sort indices by score (not scores directly)
    TokenScoreComparator comparator(d_scores_);
    thrust::sort(
        thrust::device,
        indices_ptr, indices_ptr + size_,
        comparator
    );
    
    // Allocate temporary memory for sorted arrays
    size_t bytes_float = size_ * sizeof(float);
    size_t bytes_int = size_ * sizeof(int);
    
    float* d_sorted_scores = static_cast<float*>(workspace_->Allocate(bytes_float));
    int* d_sorted_token_ids = static_cast<int*>(workspace_->Allocate(bytes_int));
    int* d_sorted_prev_indices = static_cast<int*>(workspace_->Allocate(bytes_int));
    int* d_sorted_batch_indices = static_cast<int*>(workspace_->Allocate(bytes_int));
    
    // Copy data according to sorted indices
    int block_size = 256;
    int grid_size = (size_ + block_size - 1) / block_size;
    
    ReorderDataKernel<<<grid_size, block_size>>>(
        d_sorted_scores, d_sorted_token_ids, d_sorted_prev_indices, d_sorted_batch_indices,
        d_scores_, d_token_ids_, d_prev_indices_, d_batch_indices_,
        d_indices_, size_
    );
    
    // Copy sorted data back to original arrays
    cudaMemcpy(d_scores_, d_sorted_scores, bytes_float, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_token_ids_, d_sorted_token_ids, bytes_int, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_prev_indices_, d_sorted_prev_indices, bytes_int, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_batch_indices_, d_sorted_batch_indices, bytes_int, cudaMemcpyDeviceToDevice);
    
    // Invalidate host shadow copies
    h_scores_.clear();
    h_token_ids_.clear();
    h_prev_indices_.clear();
    h_batch_indices_.clear();
}

void BeamArray::SortByScoreBatched() {
    if (size_ <= 1) {
        return;
    }
    
    // Initialize the indices
    thrust::device_ptr<int> indices_ptr(d_indices_);
    thrust::sequence(thrust::device, indices_ptr, indices_ptr + size_);
    
    // Wrap device memory in thrust device pointers
    thrust::device_ptr<float> scores_ptr(d_scores_);
    thrust::device_ptr<int> batch_indices_ptr(d_batch_indices_);
    
    // Create custom comparator for batch-aware sorting
    BatchedTokenScoreComparator comparator(d_scores_, d_batch_indices_);
    
    // Sort indices by batch then by score in descending order
    thrust::sort(
        thrust::device,
        indices_ptr, indices_ptr + size_,
        comparator
    );
    
    // Allocate temporary memory for sorted arrays
    size_t bytes_float = size_ * sizeof(float);
    size_t bytes_int = size_ * sizeof(int);
    
    float* d_sorted_scores = static_cast<float*>(workspace_->Allocate(bytes_float));
    int* d_sorted_token_ids = static_cast<int*>(workspace_->Allocate(bytes_int));
    int* d_sorted_prev_indices = static_cast<int*>(workspace_->Allocate(bytes_int));
    int* d_sorted_batch_indices = static_cast<int*>(workspace_->Allocate(bytes_int));
    
    // Copy data according to sorted indices
    int block_size = 256;
    int grid_size = (size_ + block_size - 1) / block_size;
    
    ReorderDataKernel<<<grid_size, block_size>>>(
        d_sorted_scores, d_sorted_token_ids, d_sorted_prev_indices, d_sorted_batch_indices,
        d_scores_, d_token_ids_, d_prev_indices_, d_batch_indices_,
        d_indices_, size_
    );
    
    // Copy sorted data back to original arrays
    cudaMemcpy(d_scores_, d_sorted_scores, bytes_float, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_token_ids_, d_sorted_token_ids, bytes_int, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_prev_indices_, d_sorted_prev_indices, bytes_int, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_batch_indices_, d_sorted_batch_indices, bytes_int, cudaMemcpyDeviceToDevice);
    
    // Invalidate host shadow copies
    h_scores_.clear();
    h_token_ids_.clear();
    h_prev_indices_.clear();
    h_batch_indices_.clear();
}

void BeamArray::Prune(size_t beam_width) {
    if (size_ <= beam_width) {
        return;
    }
    
    // First sort by score
    SortByScore();
    
    // Update size
    size_ = beam_width;
    
    // Recalculate batch token counts
    std::fill(batch_token_counts_.begin(), batch_token_counts_.end(), 0);
    
    // Copy batch indices to host to recount
    std::vector<int> batch_idxs(size_);
    cudaMemcpy(batch_idxs.data(), d_batch_indices_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Recount tokens per batch
    for (size_t i = 0; i < size_; i++) {
        int batch_idx = batch_idxs[i];
        if (batch_idx >= 0 && batch_idx < static_cast<int>(max_batch_size_)) {
            batch_token_counts_[batch_idx]++;
        }
    }
}

void BeamArray::PruneBatched(size_t beam_width) {
    if (size_ <= beam_width * max_batch_size_) {
        return;
    }
    
    // First sort by batch index and score
    SortByScoreBatched();
    
    // Create new temporary arrays for pruned data
    size_t max_pruned_size = beam_width * max_batch_size_;
    size_t bytes_float = max_pruned_size * sizeof(float);
    size_t bytes_int = max_pruned_size * sizeof(int);
    
    float* d_pruned_scores = static_cast<float*>(workspace_->Allocate(bytes_float));
    int* d_pruned_token_ids = static_cast<int*>(workspace_->Allocate(bytes_int));
    int* d_pruned_prev_indices = static_cast<int*>(workspace_->Allocate(bytes_int));
    int* d_pruned_batch_indices = static_cast<int*>(workspace_->Allocate(bytes_int));
    
    // Copy batch indices to host to process pruning
    std::vector<int> batch_idxs(size_);
    cudaMemcpy(batch_idxs.data(), d_batch_indices_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Reset batch token counts
    std::fill(batch_token_counts_.begin(), batch_token_counts_.end(), 0);
    
    // Process batches and copy top beam_width tokens per batch
    size_t pruned_size = 0;
    int current_batch = -1;
    size_t batch_count = 0;
    
    for (size_t i = 0; i < size_; i++) {
        int batch_idx = batch_idxs[i];
        
        // Skip invalid batch indices
        if (batch_idx < 0 || batch_idx >= static_cast<int>(max_batch_size_)) {
            continue;
        }
        
        // If we're now in a new batch, reset the counter
        if (batch_idx != current_batch) {
            current_batch = batch_idx;
            batch_count = 0;
        }
        
        // Only copy if we're still under beam_width for this batch
        if (batch_count < beam_width) {
            // Copy token to pruned array
            cudaMemcpy(d_pruned_scores + pruned_size, d_scores_ + i, sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_pruned_token_ids + pruned_size, d_token_ids_ + i, sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_pruned_prev_indices + pruned_size, d_prev_indices_ + i, sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_pruned_batch_indices + pruned_size, d_batch_indices_ + i, sizeof(int), cudaMemcpyDeviceToDevice);
            
            // Update counters
            batch_count++;
            batch_token_counts_[batch_idx]++;
            pruned_size++;
        }
    }
    
    // Copy pruned data back to original arrays
    cudaMemcpy(d_scores_, d_pruned_scores, pruned_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_token_ids_, d_pruned_token_ids, pruned_size * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_prev_indices_, d_pruned_prev_indices, pruned_size * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_batch_indices_, d_pruned_batch_indices, pruned_size * sizeof(int), cudaMemcpyDeviceToDevice);
    
    // Update size
    size_ = pruned_size;
    
    // Invalidate host shadow copies
    h_scores_.clear();
    h_token_ids_.clear();
    h_prev_indices_.clear();
    h_batch_indices_.clear();
}

Token BeamArray::GetToken(size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("Token index out of range");
    }
    
    // If we have host shadow copies, use them
    if (index < h_scores_.size()) {
        return Token(h_scores_[index], h_token_ids_[index], h_prev_indices_[index], h_batch_indices_[index]);
    }
    
    // Otherwise get from device
    Token token;
    cudaMemcpy(&token.score, d_scores_ + index, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&token.token_id, d_token_ids_ + index, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&token.prev_index, d_prev_indices_ + index, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&token.batch_index, d_batch_indices_ + index, sizeof(int), cudaMemcpyDeviceToHost);
    
    return token;
}

void BeamArray::CopyToHost(std::vector<Token>& host_tokens) const {
    // Resize host vector
    host_tokens.resize(size_);
    
    if (size_ == 0) {
        return;
    }
    
    // Allocate device memory for AoS format
    Token* d_tokens = static_cast<Token*>(workspace_->Allocate(size_ * sizeof(Token)));
    
    // Convert from SoA to AoS
    int block_size = 256;
    int grid_size = (size_ + block_size - 1) / block_size;
    convertToAoSKernel<<<grid_size, block_size>>>(d_scores_, d_token_ids_, d_prev_indices_, d_batch_indices_, d_tokens, size_);
    
    // Copy to host
    cudaMemcpy(host_tokens.data(), d_tokens, size_ * sizeof(Token), cudaMemcpyDeviceToHost);
}

void BeamArray::CopyToHostAsync(std::vector<Token>& host_tokens, cudaStream_t stream) const {
    // Resize host vector
    host_tokens.resize(size_);
    
    if (size_ == 0) {
        return;
    }
    
    // Allocate device memory for AoS format
    Token* d_tokens = static_cast<Token*>(workspace_->Allocate(size_ * sizeof(Token)));
    
    // Convert from SoA to AoS
    int block_size = 256;
    int grid_size = (size_ + block_size - 1) / block_size;
    convertToAoSKernel<<<grid_size, block_size, 0, stream>>>(d_scores_, d_token_ids_, d_prev_indices_, d_batch_indices_, d_tokens, size_);
    
    // Copy to host asynchronously
    cudaMemcpyAsync(host_tokens.data(), d_tokens, size_ * sizeof(Token), cudaMemcpyDeviceToHost, stream);
}

bool BeamArray::CopyToDevice(Token* device_tokens, cudaStream_t stream) const {
    if (size_ == 0 || device_tokens == nullptr) {
        return false;
    }
    
    // Convert from SoA to AoS
    int block_size = 256;
    int grid_size = (size_ + block_size - 1) / block_size;
    convertToAoSKernel<<<grid_size, block_size, 0, stream>>>(d_scores_, d_token_ids_, d_prev_indices_, d_batch_indices_, device_tokens, size_);
    
    return true;
}

void BeamArray::CopyToHost(std::vector<float>& scores, std::vector<int>& token_ids, std::vector<int>& prev_indices, std::vector<int>& batch_indices) {
    scores.resize(size_);
    token_ids.resize(size_);
    prev_indices.resize(size_);
    batch_indices.resize(size_);
    
    if (size_ == 0) {
        return;
    }
    
    // Copy from device to host
    cudaMemcpy(scores.data(), d_scores_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(token_ids.data(), d_token_ids_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(prev_indices.data(), d_prev_indices_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(batch_indices.data(), d_batch_indices_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Update host shadow copies
    h_scores_ = scores;
    h_token_ids_ = token_ids;
    h_prev_indices_ = prev_indices;
    h_batch_indices_ = batch_indices;
}

void BeamArray::CopyToHostAsync(std::vector<float>& scores, std::vector<int>& token_ids, std::vector<int>& prev_indices, std::vector<int>& batch_indices, cudaStream_t stream) {
    scores.resize(size_);
    token_ids.resize(size_);
    prev_indices.resize(size_);
    batch_indices.resize(size_);
    
    if (size_ == 0) {
        return;
    }
    
    // Copy from device to host asynchronously
    cudaMemcpyAsync(scores.data(), d_scores_, size_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(token_ids.data(), d_token_ids_, size_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(prev_indices.data(), d_prev_indices_, size_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(batch_indices.data(), d_batch_indices_, size_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    // No update to host shadow copies as the async copy is not complete yet
}

} // namespace beam_search
} // namespace whisper 