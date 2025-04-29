#include "whisper/beam_search/beam_array.h"
#include "whisper/beam_search/cuda_utils.h"  // For CUDA_CHECK and LAUNCH_AND_CHECK macros
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <stdexcept>

namespace whisper {
namespace beam_search {

// CUDA kernel for generating indices for sorting
__global__ void generate_indices_kernel(int* indices, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        indices[idx] = idx;
    }
}

// CUDA kernel for batched token addition
__global__ void add_tokens_kernel(float* d_scores, int* d_token_ids, int* d_prev_indices, 
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
__global__ void reorder_data_kernel(float* d_sorted_scores, int* d_sorted_token_ids, int* d_sorted_prev_indices,
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

BeamArray::BeamArray(std::size_t maxBeamSize, BeamSearchWorkspace* workspace)
    : capacity_(maxBeamSize), workspace_(workspace) {
    // Round up capacity to alignment boundary
    capacity_ = (capacity_ + kAlignment - 1) / kAlignment * kAlignment;
    allocateMemory();
    reset();
}

BeamArray::~BeamArray() {
    // Memory managed by workspace
}

// Add size() and capacity() implementations
std::size_t BeamArray::size() const {
    return size_;
}

std::size_t BeamArray::capacity() const {
    return capacity_;
}

void BeamArray::reset() {
    size_ = 0;
    
    hostScores_.clear();
    hostTokenIds_.clear();
    hostPrevIndices_.clear();
}

void BeamArray::allocateMemory() {
    size_t token_size = capacity_ * sizeof(float);
    size_t index_size = capacity_ * sizeof(int);
    
    deviceScores_ = static_cast<float*>(workspace_->allocate(token_size, kAlignment));
    deviceTokenIds_ = static_cast<int*>(workspace_->allocate(index_size, kAlignment));
    devicePrevIndices_ = static_cast<int*>(workspace_->allocate(index_size, kAlignment));
    deviceIndices_ = static_cast<int*>(workspace_->allocate(index_size, kAlignment));
    
    if (!deviceScores_ || !deviceTokenIds_ || !devicePrevIndices_ || !deviceIndices_) {
        throw std::runtime_error("Failed to allocate device memory for BeamArray");
    }
    
    hostScores_.reserve(capacity_);
    hostTokenIds_.reserve(capacity_);
    hostPrevIndices_.reserve(capacity_);
}

void BeamArray::ensureCapacity(std::size_t requiredSize) {
    if (requiredSize <= capacity_) {
        return;
    }
    
    size_t new_capacity = std::max(requiredSize, capacity_ * 2);
    new_capacity = (new_capacity + kAlignment - 1) / kAlignment * kAlignment;
    
    float* oldScores = deviceScores_;
    int* oldTokenIds = deviceTokenIds_;
    int* oldPrevIndices = devicePrevIndices_;
    
    size_t new_token_size = new_capacity * sizeof(float);
    size_t new_index_size = new_capacity * sizeof(int);
    
    deviceScores_ = static_cast<float*>(workspace_->allocate(new_token_size, kAlignment));
    deviceTokenIds_ = static_cast<int*>(workspace_->allocate(new_index_size, kAlignment));
    devicePrevIndices_ = static_cast<int*>(workspace_->allocate(new_index_size, kAlignment));
    deviceIndices_ = static_cast<int*>(workspace_->allocate(new_index_size, kAlignment));
    
    if (!deviceScores_ || !deviceTokenIds_ || !devicePrevIndices_ || !deviceIndices_) {
        throw std::runtime_error("Failed to reallocate device memory for BeamArray");
    }
    
    // Copy old data to new memory
    if (size_ > 0) {
        CUDA_CHECK(cudaMemcpy(deviceScores_, oldScores, size_ * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(deviceTokenIds_, oldTokenIds, size_ * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(devicePrevIndices_, oldPrevIndices, size_ * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    
    capacity_ = new_capacity;
    
    hostScores_.reserve(capacity_);
    hostTokenIds_.reserve(capacity_);
    hostPrevIndices_.reserve(capacity_);
}

int BeamArray::addToken(const Token& token) {
    if (size_ >= capacity_) {
        // Try to increase capacity
        ensureCapacity(size_ + 1);
        if (size_ >= capacity_) {
            return -1;  // Failed to increase capacity
        }
    }
    
    // Copy token to device memory
    CUDA_CHECK(cudaMemcpy(deviceScores_ + size_, &token.score, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceTokenIds_ + size_, &token.tokenId, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePrevIndices_ + size_, &token.prevIndex, sizeof(int), cudaMemcpyHostToDevice));
    
    // Add to host shadow copies for quick access
    hostScores_.push_back(token.score);
    hostTokenIds_.push_back(token.tokenId);
    hostPrevIndices_.push_back(token.prevIndex);
    
    return size_++;
}

int BeamArray::addTokens(const Token* tokens, std::size_t count) {
    if (count == 0) {
        return 0;
    }
    
    if (size_ + count > capacity_) {
        ensureCapacity(size_ + count);
        if (size_ + count > capacity_) {
            count = capacity_ - size_;  // Truncate to available space
            if (count == 0) {
                return 0;  // No space available
            }
        }
    }
    
    std::vector<float> newScores(count);
    std::vector<int> newTokenIds(count);
    std::vector<int> newPrevIndices(count);
    
    for (std::size_t i = 0; i < count; i++) {
        newScores[i] = tokens[i].score;
        newTokenIds[i] = tokens[i].tokenId;
        newPrevIndices[i] = tokens[i].prevIndex;
        
        hostScores_.push_back(tokens[i].score);
        hostTokenIds_.push_back(tokens[i].tokenId);
        hostPrevIndices_.push_back(tokens[i].prevIndex);
    }
    
    float* deviceNewScores;
    int* deviceNewTokenIds;
    int* deviceNewPrevIndices;
    
    CUDA_CHECK(cudaMalloc(&deviceNewScores, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceNewTokenIds, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceNewPrevIndices, count * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(deviceNewScores, newScores.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceNewTokenIds, newTokenIds.data(), count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceNewPrevIndices, newPrevIndices.data(), count * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel to add tokens in parallel
    int blocks = (count + kBlockSize - 1) / kBlockSize;
    LAUNCH_AND_CHECK(add_tokens_kernel<<<blocks, kBlockSize>>>(
        deviceScores_, deviceTokenIds_, devicePrevIndices_,
        deviceNewScores, deviceNewTokenIds, deviceNewPrevIndices,
        size_, count
    ));
    
    CUDA_CHECK(cudaFree(deviceNewScores));
    CUDA_CHECK(cudaFree(deviceNewTokenIds));
    CUDA_CHECK(cudaFree(deviceNewPrevIndices));
    
    size_ += count;
    return count;
}

void BeamArray::sortByScore() {
    if (size_ <= 1) {
        return;  // Nothing to sort
    }
    
    // Generate indices for sorting
    int blocks = (size_ + kBlockSize - 1) / kBlockSize;
    LAUNCH_AND_CHECK(generate_indices_kernel<<<blocks, kBlockSize>>>(
        deviceIndices_, size_
    ));
    
    // Sort indices based on scores (descending order)
    thrust::device_ptr<int> deviceIndicesPtr(deviceIndices_);
    thrust::sort(thrust::device, deviceIndicesPtr, deviceIndicesPtr + size_, 
                TokenScoreComparator(deviceScores_));
    
    // Allocate temporary memory for sorted data
    float* sortedScores;
    int* sortedTokenIds;
    int* sortedPrevIndices;
    
    CUDA_CHECK(cudaMalloc(&sortedScores, size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sortedTokenIds, size_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sortedPrevIndices, size_ * sizeof(int)));
    
    // Reorder data in parallel based on sorted indices
    LAUNCH_AND_CHECK(reorder_data_kernel<<<blocks, kBlockSize>>>(
        sortedScores, sortedTokenIds, sortedPrevIndices,
        deviceScores_, deviceTokenIds_, devicePrevIndices_,
        deviceIndices_, size_
    ));
    
    // Copy sorted data back
    CUDA_CHECK(cudaMemcpy(deviceScores_, sortedScores, size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(deviceTokenIds_, sortedTokenIds, size_ * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(devicePrevIndices_, sortedPrevIndices, size_ * sizeof(int), cudaMemcpyDeviceToDevice));
    
    CUDA_CHECK(cudaFree(sortedScores));
    CUDA_CHECK(cudaFree(sortedTokenIds));
    CUDA_CHECK(cudaFree(sortedPrevIndices));
    
    // Update host shadow copies
    copyToHost(hostScores_, hostTokenIds_, hostPrevIndices_);
}

void BeamArray::prune(std::size_t beamWidth) {
    if (size_ <= beamWidth) {
        return;  // No pruning needed
    }
    
    // Sort first to ensure we keep the best tokens
    sortByScore();
    
    // Truncate to beam width
    size_ = beamWidth;
    
    // Truncate host shadow copies
    hostScores_.resize(size_);
    hostTokenIds_.resize(size_);
    hostPrevIndices_.resize(size_);
}

Token BeamArray::getToken(std::size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("Token index out of range");
    }
    
    // Use host shadow copies for fast access if available
    if (index < hostScores_.size()) {
        return Token(hostScores_[index], hostTokenIds_[index], hostPrevIndices_[index]);
    }
    
    // Otherwise, fetch from device
    Token token;
    CUDA_CHECK(cudaMemcpy(&token.score, deviceScores_ + index, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&token.tokenId, deviceTokenIds_ + index, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&token.prevIndex, devicePrevIndices_ + index, sizeof(int), cudaMemcpyDeviceToHost));
    
    return token;
}

void BeamArray::copyToHost(std::vector<Token>& hostTokens) const {
    hostTokens.resize(size_);
    
    if (size_ == 0) {
        return;
    }
    
    std::vector<float> scores(size_);
    std::vector<int> tokenIds(size_);
    std::vector<int> prevIndices(size_);
    
    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(scores.data(), deviceScores_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tokenIds.data(), deviceTokenIds_, size_ * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(prevIndices.data(), devicePrevIndices_, size_ * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Construct tokens
    for (std::size_t i = 0; i < size_; i++) {
        hostTokens[i] = Token(scores[i], tokenIds[i], prevIndices[i]);
    }
}

void BeamArray::copyToHost(std::vector<float>& scores, std::vector<int>& tokenIds, std::vector<int>& prevIndices) {
    scores.resize(size_);
    tokenIds.resize(size_);
    prevIndices.resize(size_);
    
    if (size_ == 0) {
        return;
    }
    
    CUDA_CHECK(cudaMemcpy(scores.data(), deviceScores_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tokenIds.data(), deviceTokenIds_, size_ * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(prevIndices.data(), devicePrevIndices_, size_ * sizeof(int), cudaMemcpyDeviceToHost));
}

} // namespace beam_search
} // namespace whisper 