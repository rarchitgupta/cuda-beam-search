#include "whisper/beam_search/logit_processor.h"
#include "whisper/beam_search/cuda_utils.h"  // For CUDA_CHECK/LAUNCH_AND_CHECK
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace whisper {
namespace beam_search {

// CUDA kernels

// Apply temperature to logits
__global__ void temperature_kernel(float* logits, size_t size, float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        logits[idx] = logits[idx] / temperature;
    }
}

// Find max value for stable softmax
__global__ void logit_max_kernel(const float* logits, float* max_values, int vocab_size) {
    int batch_pos = blockIdx.x;
    int offset = batch_pos * vocab_size;
    
    // Using shared memory for reduction
    __shared__ float shared_max[256];
    
    // Initialize with first value
    float max_val = logits[offset];
    
    // Find max across vocabulary
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        max_val = max(max_val, logits[offset + i]);
    }
    
    // Store thread's max to shared memory
    shared_max[threadIdx.x] = max_val;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = max(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    // Write result
    if (threadIdx.x == 0) {
        max_values[batch_pos] = shared_max[0];
    }
}

// Compute softmax efficiently
__global__ void softmax_kernel(
    const float* logits, float* probs, const float* max_values, 
    int vocab_size, float* sum_values) {
    
    int batch_pos = blockIdx.x;
    int offset = batch_pos * vocab_size;
    float max_val = max_values[batch_pos];
    
    // Using shared memory for sum reduction
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = 0.0f;
    
    // Compute exp(logit - max) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = expf(logits[offset + i] - max_val);
        probs[offset + i] = val;
        thread_sum += val;
    }
    
    // Store thread's sum to shared memory
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write sum result
    if (threadIdx.x == 0) {
        sum_values[batch_pos] = shared_sum[0];
    }
}

// Normalize values after softmax
__global__ void normalize_kernel(float* probs, const float* sum_values, int vocab_size) {
    int batch_pos = blockIdx.x;
    int offset = batch_pos * vocab_size;
    float sum = sum_values[batch_pos];
    
    // Normalize each probability
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        probs[offset + i] /= sum;
    }
}

// Generate indices for sorting
__global__ void init_indices_kernel(int* indices, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        indices[idx] = idx;
    }
}

// Score tokens kernel
__global__ void score_tokens_kernel(
    const float* logits, const float* prev_scores, int* token_ids, int* prev_indices,
    float* new_scores, int* new_token_ids, int* new_prev_indices,
    int beam_size, int vocab_size) {
    
    int beam_idx = blockIdx.y;
    int vocab_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (beam_idx < beam_size && vocab_idx < vocab_size) {
        int out_idx = beam_idx * vocab_size + vocab_idx;
        
        // Combine scores
        new_scores[out_idx] = prev_scores[beam_idx] + logits[vocab_idx];
        new_token_ids[out_idx] = vocab_idx;
        new_prev_indices[out_idx] = beam_idx;
    }
}

// Top-K comparator for sorting
struct ScoreComparator {
    const float* scores;
    
    ScoreComparator(const float* s) : scores(s) {}
    
    __host__ __device__ bool operator()(int a, int b) const {
        return scores[a] > scores[b]; // Descending order
    }
};

// Constants
constexpr int kBlockSize = 256;
constexpr size_t kAlignment = 128; // Align to 128 bytes

// Implementation

LogitProcessor::LogitProcessor(
    BeamSearchWorkspace* workspace, float temperature, int topK, float topP)
    : workspace_(workspace), temperature_(temperature), topK_(topK), topP_(topP) {
    
    if (!workspace_) {
        throw std::invalid_argument("Workspace cannot be null");
    }
}

bool LogitProcessor::processLogits(
    const float* deviceLogits, int batchSize, int seqLen, int vocabSize) {
    
    // Store dimensions
    batchSize_ = batchSize;
    seqLen_ = seqLen;
    vocabSize_ = vocabSize;
    
    // Calculate required storage size
    size_t logitsSize = batchSize_ * seqLen_ * vocabSize_ * sizeof(float);
    size_t indicesSize = batchSize_ * seqLen_ * vocabSize_ * sizeof(int);
    size_t tempSize = batchSize_ * seqLen_ * sizeof(float) * 2; // For max and sum values
    
    // Total storage needed
    size_t totalSize = logitsSize + indicesSize + tempSize;
    
    // Allocate memory if needed
    allocateMemory(totalSize);
    
    // Copy logits to our processed buffer
    CUDA_CHECK(cudaMemcpy(processedLogits_, deviceLogits, logitsSize, cudaMemcpyDeviceToDevice));
    
    return true;
}

void LogitProcessor::allocateMemory(std::size_t requiredSize) {
    // Only reallocate if needed
    if (requiredSize <= tempStorageSize_ && processedLogits_ != nullptr) {
        return;
    }
    
    // Allocate with proper alignment
    tempStorageSize_ = requiredSize + kAlignment;
    processedLogits_ = static_cast<float*>(workspace_->allocate(
        batchSize_ * seqLen_ * vocabSize_ * sizeof(float), kAlignment));
    
    tokenIndices_ = static_cast<int*>(workspace_->allocate(
        batchSize_ * seqLen_ * vocabSize_ * sizeof(int), kAlignment));
    
    tempStorage_ = static_cast<float*>(workspace_->allocate(
        batchSize_ * seqLen_ * sizeof(float) * 2, kAlignment));
    
    if (!processedLogits_ || !tokenIndices_ || !tempStorage_) {
        throw std::runtime_error("Failed to allocate device memory for LogitProcessor");
    }
}

void LogitProcessor::applyTemperature(float* deviceLogits, int batchIndex, int position) {
    int offset = (batchIndex * seqLen_ + position) * vocabSize_;
    int blocks = (vocabSize_ + kBlockSize - 1) / kBlockSize;
    
    LAUNCH_AND_CHECK(temperature_kernel<<<blocks, kBlockSize>>>(
        deviceLogits + offset, vocabSize_, temperature_
    ));
}

void LogitProcessor::applySoftmax(float* deviceLogits, int batchIndex, int position) {
    int batchPos = batchIndex * seqLen_ + position;
    float* d_maxValues = tempStorage_;
    float* d_sumValues = tempStorage_ + batchSize_ * seqLen_;
    
    // Find max for numerical stability
    LAUNCH_AND_CHECK(logit_max_kernel<<<1, kBlockSize>>>(
        deviceLogits + batchPos * vocabSize_, d_maxValues + batchPos, vocabSize_
    ));
    
    // Compute softmax
    LAUNCH_AND_CHECK(softmax_kernel<<<1, kBlockSize>>>(
        deviceLogits + batchPos * vocabSize_, 
        deviceLogits + batchPos * vocabSize_,
        d_maxValues + batchPos, 
        vocabSize_,
        d_sumValues + batchPos
    ));
    
    // Normalize
    LAUNCH_AND_CHECK(normalize_kernel<<<1, kBlockSize>>>(
        deviceLogits + batchPos * vocabSize_, 
        d_sumValues + batchPos, 
        vocabSize_
    ));
}

void LogitProcessor::applyTopK(int batchIndex, int position) {
    if (topK_ <= 0 || topK_ >= vocabSize_) {
        return; // No need to apply top-k
    }
    
    int batchPos = batchIndex * seqLen_ + position;
    int offset = batchPos * vocabSize_;
    int blocks = (vocabSize_ + kBlockSize - 1) / kBlockSize;
    
    // Initialize indices
    LAUNCH_AND_CHECK(init_indices_kernel<<<blocks, kBlockSize>>>(
        tokenIndices_ + offset, vocabSize_
    ));
    
    // Sort indices by score
    thrust::device_ptr<int> d_indicesPtr(tokenIndices_ + offset);
    thrust::sort(
        thrust::device, 
        d_indicesPtr, 
        d_indicesPtr + vocabSize_,
        ScoreComparator(processedLogits_ + offset));
    
    // Only keep top-k logits (set others to -INFINITY)
    float negInf = -std::numeric_limits<float>::infinity();
    
    // To keep implementation simple for now, we'll do this on CPU
    // A more optimized version would use a custom kernel
    std::vector<int> h_indices(vocabSize_);
    CUDA_CHECK(cudaMemcpy(h_indices.data(), tokenIndices_ + offset, 
              vocabSize_ * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::vector<float> h_logits(vocabSize_);
    CUDA_CHECK(cudaMemcpy(h_logits.data(), processedLogits_ + offset, 
              vocabSize_ * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Set logits outside top-k to -inf
    for (int i = topK_; i < vocabSize_; i++) {
        h_logits[h_indices[i]] = negInf;
    }
    
    // Copy back to device
    CUDA_CHECK(cudaMemcpy(processedLogits_ + offset, h_logits.data(), 
              vocabSize_ * sizeof(float), cudaMemcpyHostToDevice));
}

void LogitProcessor::applyTopP(int batchIndex, int position) {
    if (topP_ >= 1.0f - 1e-6) return;
    // TODO: Replace this CPU fallback with a CUDA kernel
    
    // int batchPos = batchIndex * seqLen_ + position;
    // int offset = batchPos * vocabSize_;
    // Fallback code removed. Will implement GPU version later.
}

void LogitProcessor::scoreNextTokens(
    const BeamArray* beam, int batchIndex, int position, BeamArray* outputBeam) {
    
    // Process logits for this position
    int batchPos = batchIndex * seqLen_ + position;
    int offset = batchPos * vocabSize_;
    
    // Apply temperature and softmax
    applyTemperature(processedLogits_, batchIndex, position);
    applySoftmax(processedLogits_, batchIndex, position);
    
    // Apply top-k and top-p sampling if enabled
    applyTopK(batchIndex, position);
    applyTopP(batchIndex, position);
    
    // Now score tokens based on beam and processed logits
    size_t beamSize = beam->size();
    size_t outputSize = beamSize * vocabSize_;
    
    // Allocate temporary memory for expanded tokens
    float* d_expandedScores = static_cast<float*>(workspace_->allocate(
        outputSize * sizeof(float), kAlignment));
    int* d_expandedTokenIds = static_cast<int*>(workspace_->allocate(
        outputSize * sizeof(int), kAlignment));
    int* d_expandedPrevIndices = static_cast<int*>(workspace_->allocate(
        outputSize * sizeof(int), kAlignment));
    
    // Get beam data pointers
    float* d_beamScores = beam->scorePtr();
    int* d_beamTokenIds = beam->tokenIdPtr();
    int* d_beamPrevIndices = beam->prevIndexPtr();
    
    // Launch kernel to score tokens
    dim3 blockDim(kBlockSize);
    dim3 gridDim((vocabSize_ + blockDim.x - 1) / blockDim.x, beamSize);
    
    LAUNCH_AND_CHECK(score_tokens_kernel<<<gridDim, blockDim>>>(
        processedLogits_ + offset,
        d_beamScores, 
        d_beamTokenIds,
        d_beamPrevIndices,
        d_expandedScores,
        d_expandedTokenIds,
        d_expandedPrevIndices,
        beamSize,
        vocabSize_
    ));
    
    // Create expanded tokens and add to output beam
    std::vector<Token> tokens(outputSize);
    
    // Copy expanded tokens to host
    std::vector<float> h_scores(outputSize);
    std::vector<int> h_tokenIds(outputSize);
    std::vector<int> h_prevIndices(outputSize);
    
    CUDA_CHECK(cudaMemcpy(h_scores.data(), d_expandedScores, 
              outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tokenIds.data(), d_expandedTokenIds, 
              outputSize * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_prevIndices.data(), d_expandedPrevIndices, 
              outputSize * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Create tokens
    for (size_t i = 0; i < outputSize; i++) {
        tokens[i] = Token(h_scores[i], h_tokenIds[i], h_prevIndices[i]);
    }
    
    // Add expanded tokens to output beam
    outputBeam->addTokens(tokens.data(), outputSize);
}

void LogitProcessor::scoreAndPrune(
    const BeamArray* beam, int batchIndex, int position, 
    BeamArray* outputBeam, size_t beamWidth) {
    
    // Score tokens
    scoreNextTokens(beam, batchIndex, position, outputBeam);
    
    // Prune beam
    outputBeam->prune(beamWidth);
}

void LogitProcessor::setSamplingParams(float temperature, int topK, float topP) {
    temperature_ = temperature;
    topK_ = topK;
    topP_ = topP;
}

} // namespace beam_search
} // namespace whisper 