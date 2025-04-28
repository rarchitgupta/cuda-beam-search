#include "whisper/beam_search/logit_processor.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace whisper {
namespace beam_search {

// Simplified CUDA error checking
inline bool checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << message << " - " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

// CUDA kernels

// Convert half precision to float
__global__ void HalfToFloatKernel(const __half* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

// Apply temperature to logits
__global__ void TemperatureKernel(float* logits, size_t size, float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        logits[idx] = logits[idx] / temperature;
    }
}

// Find max value for stable softmax
__global__ void LogitMaxKernel(const float* logits, float* max_values, int vocab_size) {
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
__global__ void SoftmaxKernel(
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
__global__ void NormalizeKernel(float* probs, const float* sum_values, int vocab_size) {
    int batch_pos = blockIdx.x;
    int offset = batch_pos * vocab_size;
    float sum = sum_values[batch_pos];
    
    // Normalize each probability
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        probs[offset + i] /= sum;
    }
}

// Generate indices for sorting
__global__ void InitIndicesKernel(int* indices, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        indices[idx] = idx;
    }
}

// Score tokens kernel
__global__ void ScoreTokensKernel(
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
    BeamSearchWorkspace* workspace, float temperature, int top_k, float top_p,
    utils::CudaStreamManager* stream_manager)
    : workspace_(workspace), temperature_(temperature), top_k_(top_k), top_p_(top_p),
      stream_manager_(stream_manager), owns_stream_manager_(false) {
    
    if (!workspace_) {
        throw std::invalid_argument("Workspace cannot be null");
    }
    
    // Create stream manager if not provided
    if (!stream_manager_) {
        stream_manager_ = new utils::CudaStreamManager();
        owns_stream_manager_ = true;
    }
}

LogitProcessor::~LogitProcessor() {
    if (owns_stream_manager_ && stream_manager_) {
        delete stream_manager_;
        stream_manager_ = nullptr;
    }
}

bool LogitProcessor::ProcessLogits(
    const float* d_logits, int batch_size, int seq_len, int vocab_size) {
    
    // Validate input dimensions
    if (batch_size <= 0 || seq_len <= 0 || vocab_size <= 0) {
        std::cerr << "Invalid dimensions: batch_size=" << batch_size 
                  << ", seq_len=" << seq_len << ", vocab_size=" << vocab_size << std::endl;
        return false;
    }
    
    // Store dimensions
    batch_size_ = batch_size;
    seq_len_ = seq_len;
    vocab_size_ = vocab_size;
    
    // Calculate required storage size
    size_t logits_size = batch_size * seq_len * vocab_size * sizeof(float);
    size_t indices_size = batch_size * seq_len * vocab_size * sizeof(int);
    size_t temp_size = batch_size * seq_len * sizeof(float) * 2; // For max and sum values
    
    // Total storage needed
    size_t total_size = logits_size + indices_size + temp_size;
    
    // Allocate memory if needed
    AllocateMemory(total_size);
    
    // Copy logits to our processed buffer using compute stream
    cudaStream_t stream = GetComputeStream();
    
    cudaError_t error = cudaMemcpyAsync(d_processed_logits_, d_logits, logits_size, 
                                        cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
        std::cerr << "ProcessLogits - cudaMemcpyAsync error: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

bool LogitProcessor::ProcessLogitsHalf(
    const void* d_logits, int batch_size, int seq_len, int vocab_size) {
    
    // Validate input dimensions
    if (batch_size <= 0 || seq_len <= 0 || vocab_size <= 0) {
        std::cerr << "Invalid dimensions: batch_size=" << batch_size 
                  << ", seq_len=" << seq_len << ", vocab_size=" << vocab_size << std::endl;
        return false;
    }
    
    // Store dimensions
    batch_size_ = batch_size;
    seq_len_ = seq_len;
    vocab_size_ = vocab_size;
    
    // Calculate required storage size
    size_t logits_size = batch_size * seq_len * vocab_size * sizeof(float);
    size_t indices_size = batch_size * seq_len * vocab_size * sizeof(int);
    size_t temp_size = batch_size * seq_len * sizeof(float) * 2; // For max and sum values
    
    // Total storage needed
    size_t total_size = logits_size + indices_size + temp_size;
    
    // Allocate memory if needed
    AllocateMemory(total_size);
    
    // Convert half to float using compute stream
    const int total_elements = batch_size * seq_len * vocab_size;
    const int blockSize = 256;
    const int gridSize = (total_elements + blockSize - 1) / blockSize;
    
    cudaStream_t stream = GetComputeStream();
    HalfToFloatKernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const __half*>(d_logits), 
        d_processed_logits_,
        total_elements);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "HalfToFloatKernel error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

void LogitProcessor::AllocateMemory(size_t required_size) {
    // Only reallocate if needed
    if (required_size <= temp_storage_size_ && d_processed_logits_ != nullptr) {
        return;
    }
    
    // Allocate with proper alignment
    temp_storage_size_ = required_size + kAlignment;
    d_processed_logits_ = static_cast<float*>(workspace_->Allocate(
        batch_size_ * seq_len_ * vocab_size_ * sizeof(float), kAlignment));
    
    d_token_indices_ = static_cast<int*>(workspace_->Allocate(
        batch_size_ * seq_len_ * vocab_size_ * sizeof(int), kAlignment));
    
    d_temp_storage_ = static_cast<float*>(workspace_->Allocate(
        batch_size_ * seq_len_ * sizeof(float) * 2, kAlignment));
    
    if (!d_processed_logits_ || !d_token_indices_ || !d_temp_storage_) {
        throw std::runtime_error("Failed to allocate device memory for LogitProcessor");
    }
}

void LogitProcessor::ApplyTemperature(float* d_logits, int batch_index, int position) {
    int offset = (batch_index * seq_len_ + position) * vocab_size_;
    int blocks = (vocab_size_ + kBlockSize - 1) / kBlockSize;
    
    // Validate kernel launch parameters
    if (blocks <= 0 || vocab_size_ <= 0) {
        throw std::runtime_error("Invalid kernel configuration");
    }
    
    // Get the stream for this operation
    cudaStream_t stream = GetComputeStream();
    
    // Launch the temperature kernel
    TemperatureKernel<<<blocks, kBlockSize, 0, stream>>>(
        d_logits + offset, vocab_size_, temperature_);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void LogitProcessor::ApplySoftmax(float* d_logits, int batch_index, int position) {
    int batch_pos = batch_index * seq_len_ + position;
    float* d_max_values = d_temp_storage_;
    float* d_sum_values = d_temp_storage_ + batch_size_ * seq_len_;
    cudaStream_t stream = GetComputeStream();
    
    // Find max for numerical stability
    LogitMaxKernel<<<1, kBlockSize, 0, stream>>>(
        d_logits + batch_pos * vocab_size_, d_max_values + batch_pos, vocab_size_);
    
    // Compute softmax
    SoftmaxKernel<<<1, kBlockSize, 0, stream>>>(
        d_logits + batch_pos * vocab_size_, 
        d_logits + batch_pos * vocab_size_,
        d_max_values + batch_pos, 
        vocab_size_,
        d_sum_values + batch_pos);
    
    // Normalize
    NormalizeKernel<<<1, kBlockSize, 0, stream>>>(
        d_logits + batch_pos * vocab_size_, 
        d_sum_values + batch_pos, 
        vocab_size_);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ApplySoftmax: ") + cudaGetErrorString(error));
    }
}

void LogitProcessor::ApplyTopK(int batch_index, int position) {
    if (top_k_ <= 0 || top_k_ >= vocab_size_) {
        return; // No need to apply top-k
    }
    
    int batch_pos = batch_index * seq_len_ + position;
    int offset = batch_pos * vocab_size_;
    int blocks = (vocab_size_ + kBlockSize - 1) / kBlockSize;
    
    // Initialize indices
    InitIndicesKernel<<<blocks, kBlockSize>>>(
        d_token_indices_ + offset, vocab_size_);
    
    // Sort indices by score
    thrust::device_ptr<int> d_indices_ptr(d_token_indices_ + offset);
    thrust::sort(
        thrust::device, 
        d_indices_ptr, 
        d_indices_ptr + vocab_size_,
        ScoreComparator(d_processed_logits_ + offset));
    
    // Only keep top-k logits (set others to -INFINITY)
    float neg_inf = -std::numeric_limits<float>::infinity();
    
    // Simple CPU implementation for now
    std::vector<int> h_indices(vocab_size_);
    cudaMemcpy(h_indices.data(), d_token_indices_ + offset, 
              vocab_size_ * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::vector<float> h_logits(vocab_size_);
    cudaMemcpy(h_logits.data(), d_processed_logits_ + offset, 
              vocab_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Set logits outside top-k to -inf
    for (int i = top_k_; i < vocab_size_; i++) {
        h_logits[h_indices[i]] = neg_inf;
    }
    
    // Copy back to device
    cudaMemcpy(d_processed_logits_ + offset, h_logits.data(), 
              vocab_size_ * sizeof(float), cudaMemcpyHostToDevice);
}

void LogitProcessor::ApplyTopP(int batch_index, int position) {
    if (top_p_ >= 1.0f - 1e-6) {
        return; // No need to apply top-p
    }
    
    // Basic top-p implementation on CPU
    int batch_pos = batch_index * seq_len_ + position;
    int offset = batch_pos * vocab_size_;
    
    // Get sorted indices and probs
    std::vector<int> h_indices(vocab_size_);
    std::vector<float> h_probs(vocab_size_);
    
    cudaMemcpy(h_indices.data(), d_token_indices_ + offset, 
              vocab_size_ * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(h_probs.data(), d_processed_logits_ + offset, 
              vocab_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate cumulative probability
    float cum_prob = 0.0f;
    float neg_inf = -std::numeric_limits<float>::infinity();
    int cutoff_idx = vocab_size_ - 1;
    
    for (int i = 0; i < vocab_size_; i++) {
        cum_prob += h_probs[h_indices[i]];
        if (cum_prob > top_p_) {
            cutoff_idx = i;
            break;
        }
    }
    
    // Set probabilities outside top-p to 0
    for (int i = cutoff_idx + 1; i < vocab_size_; i++) {
        h_probs[h_indices[i]] = neg_inf;
    }
    
    // Copy back to device
    cudaMemcpy(d_processed_logits_ + offset, h_probs.data(), 
              vocab_size_ * sizeof(float), cudaMemcpyDeviceToHost);
}

void LogitProcessor::ScoreNextTokens(const BeamArray* beam, int position, BeamArray* output_beam, int batch_index) {
    // If batch_index is -1, process all batches
    if (batch_index == -1) {
        // Process each batch
        for (int b = 0; b < batch_size_; b++) {
            ScoreNextTokensImpl(beam, b, position, output_beam);
        }
        return;
    }
    
    // Process a single batch
    ScoreNextTokensImpl(beam, batch_index, position, output_beam);
}

// Internal implementation to process a single batch
void LogitProcessor::ScoreNextTokensImpl(const BeamArray* beam, int batch_index, int position, BeamArray* output_beam) {
    // Validate parameters before proceeding
    if (batch_index < 0 || batch_index >= batch_size_ || position < 0 || position >= seq_len_) {
        std::cerr << "ERROR: Invalid batch_index or position. batch_index=" << batch_index 
                  << ", batch_size_=" << batch_size_ << ", position=" << position 
                  << ", seq_len_=" << seq_len_ << std::endl;
        return;
    }
    
    // Process logits for this position
    int batch_pos = batch_index * seq_len_ + position;
    int offset = batch_pos * vocab_size_;
    
    try {
        // Process the logits
        ApplyTemperature(d_processed_logits_, batch_index, position);
        ApplySoftmax(d_processed_logits_, batch_index, position);
        ApplyTopK(batch_index, position);
        ApplyTopP(batch_index, position);
        
        // Now score tokens based on beam and processed logits
        size_t beam_size = beam->Size();
        size_t output_size = beam_size * vocab_size_;
        
        // Allocate temporary memory for expanded tokens
        float* d_expanded_scores = static_cast<float*>(workspace_->Allocate(
            output_size * sizeof(float), kAlignment));
        int* d_expanded_token_ids = static_cast<int*>(workspace_->Allocate(
            output_size * sizeof(int), kAlignment));
        int* d_expanded_prev_indices = static_cast<int*>(workspace_->Allocate(
            output_size * sizeof(int), kAlignment));
        
        if (!d_expanded_scores || !d_expanded_token_ids || !d_expanded_prev_indices) {
            throw std::runtime_error("Failed to allocate memory for expanded tokens");
        }
        
        // Get beam data pointers
        float* d_beam_scores = beam->GetScorePtr();
        int* d_beam_token_ids = beam->GetTokenIdPtr();
        int* d_beam_prev_indices = beam->GetPrevIndexPtr();
        
        // Launch kernel to score tokens
        dim3 block_dim(kBlockSize);
        dim3 grid_dim((vocab_size_ + block_dim.x - 1) / block_dim.x, beam_size);
        
        // Check for zero grid dimension
        if (grid_dim.x == 0 || grid_dim.y == 0) {
            std::cerr << "ERROR: Invalid grid dimensions for ScoreTokensKernel" << std::endl;
            return;
        }
        
        ScoreTokensKernel<<<grid_dim, block_dim>>>(
            d_processed_logits_ + offset,
            d_beam_scores, 
            d_beam_token_ids,
            d_beam_prev_indices,
            d_expanded_scores,
            d_expanded_token_ids,
            d_expanded_prev_indices,
            beam_size,
            vocab_size_);
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (!checkCudaError(err, "ScoreTokensKernel failed")) {
            return;
        }
        
        // Wait for kernel to complete
        cudaStreamSynchronize(GetComputeStream());
        
        // Create expanded tokens and add to output beam
        std::vector<Token> tokens(output_size);
        
        // Copy expanded tokens to host
        std::vector<float> h_scores(output_size);
        std::vector<int> h_token_ids(output_size);
        std::vector<int> h_prev_indices(output_size);
        
        cudaMemcpy(h_scores.data(), d_expanded_scores, 
                  output_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_token_ids.data(), d_expanded_token_ids, 
                  output_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_prev_indices.data(), d_expanded_prev_indices, 
                  output_size * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Create tokens
        for (size_t i = 0; i < output_size; i++) {
            tokens[i] = Token(h_scores[i], h_token_ids[i], h_prev_indices[i], batch_index);
        }
        
        // Add expanded tokens to output beam
        output_beam->AddTokens(tokens.data(), output_size);
    } catch (const std::exception& e) {
        std::cerr << "Exception in batch " << batch_index << ": " << e.what() << std::endl;
    }
}

void LogitProcessor::ScoreAndPrune(const BeamArray* beam, int position, 
                                  BeamArray* output_beam, size_t beam_width,
                                  int batch_index) {
    // Score tokens
    ScoreNextTokens(beam, position, output_beam, batch_index);
    
    // Prune beam
    output_beam->Prune(beam_width);
}

void LogitProcessor::ScoreNextTokensBatched(const BeamArray* beam, int position, BeamArray* output_beam) {
    // Process each batch
    for (int batch_index = 0; batch_index < batch_size_; batch_index++) {
        try {
            // Process this batch
            ScoreNextTokens(beam, position, output_beam, batch_index);
        } catch (const std::exception& e) {
            std::cerr << "Exception in batch " << batch_index << ": " << e.what() << std::endl;
        }
    }
}

void LogitProcessor::ScoreAndPruneBatched(const BeamArray* beam, int position,
                                         BeamArray* output_beam, size_t beam_width) {
    // Process all batches
    ScoreNextTokensBatched(beam, position, output_beam);
    
    // Prune beam
    output_beam->Prune(beam_width);
}

void LogitProcessor::SetSamplingParams(float temperature, int top_k, float top_p) {
    temperature_ = temperature;
    top_k_ = top_k;
    top_p_ = top_p;
}

utils::CudaStreamManager* LogitProcessor::GetStreamManager() const {
    return stream_manager_;
}

cudaStream_t LogitProcessor::GetComputeStream() const {
    return stream_manager_ ? stream_manager_->GetComputeStream() : nullptr;
}

// Batched functions
void LogitProcessor::ApplyTemperatureBatched(float* d_logits, int position) {
    for (int batch_index = 0; batch_index < batch_size_; batch_index++) {
        ApplyTemperature(d_logits, batch_index, position);
    }
}

void LogitProcessor::ApplySoftmaxBatched(float* d_logits, int position) {
    for (int batch_index = 0; batch_index < batch_size_; batch_index++) {
        ApplySoftmax(d_logits, batch_index, position);
    }
}

void LogitProcessor::ApplyTopKBatched(int position) {
    for (int batch_index = 0; batch_index < batch_size_; batch_index++) {
        ApplyTopK(batch_index, position);
    }
}

void LogitProcessor::ApplyTopPBatched(int position) {
    for (int batch_index = 0; batch_index < batch_size_; batch_index++) {
        ApplyTopP(batch_index, position);
    }
}

} // namespace beam_search
} // namespace whisper 