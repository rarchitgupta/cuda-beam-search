#pragma once

#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_array.h"
#include "whisper/utils/cuda_stream_manager.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace whisper {
namespace beam_search {

// Processes logits from Whisper model and applies sampling strategies (temperature, top-k, top-p)
class LogitProcessor {
public:
    // Constructor with optional stream manager
    LogitProcessor(BeamSearchWorkspace* workspace, float temperature = 1.0f, 
                  int top_k = 0, float top_p = 1.0f,
                  utils::CudaStreamManager* stream_manager = nullptr);
                  
    // Destructor to clean up owned stream manager
    ~LogitProcessor();

    // Process float32 logits
    bool ProcessLogits(const float* d_logits, int batch_size, int seq_len, int vocab_size);
    
    // Process float16 logits (automatically converts to float32)
    bool ProcessLogitsHalf(const void* d_logits, int batch_size, int seq_len, int vocab_size);

    /**
     * Score next tokens for a beam. If batch_index is -1, processes all batches.
     * 
     * @param beam The input beam array
     * @param position The position in the sequence
     * @param output_beam The output beam array to store results
     * @param batch_index The batch index (-1 to process all batches)
     */
    void ScoreNextTokens(const BeamArray* beam, int position, BeamArray* output_beam, 
                        int batch_index = -1);

    /**
     * Score tokens and prune to beam_width in one step
     * 
     * @param beam The input beam array
     * @param position The position in the sequence
     * @param output_beam The output beam array to store results
     * @param beam_width The width to prune to
     * @param batch_index The batch index (-1 to process all batches)
     */
    void ScoreAndPrune(const BeamArray* beam, int position, 
                      BeamArray* output_beam, size_t beam_width,
                      int batch_index = -1);

    // Set sampling parameters
    void SetSamplingParams(float temperature, int top_k = 0, float top_p = 1.0f);
    
    // Get stream manager or nullptr if not used
    utils::CudaStreamManager* GetStreamManager() const;

private:
    float* d_processed_logits_ = nullptr;
    int* d_token_indices_ = nullptr;
    float* d_temp_storage_ = nullptr;
    
    float temperature_ = 1.0f;
    int top_k_ = 0;
    float top_p_ = 1.0f;
    
    int batch_size_ = 0;
    int seq_len_ = 0;
    int vocab_size_ = 0;
    
    BeamSearchWorkspace* workspace_ = nullptr;
    size_t temp_storage_size_ = 0;
    
    // Stream manager (may be nullptr if not used)
    utils::CudaStreamManager* stream_manager_ = nullptr;
    bool owns_stream_manager_ = false;
    
    void AllocateMemory(size_t required_size);
    
    // Internal implementation for processing a single batch
    void ScoreNextTokensImpl(const BeamArray* beam, int batch_index, int position, BeamArray* output_beam);
    
    // Single batch processing helpers
    void ApplyTemperature(float* d_logits, int batch_index, int position);
    void ApplySoftmax(float* d_logits, int batch_index, int position);
    void ApplyTopK(int batch_index, int position);
    void ApplyTopP(int batch_index, int position);
    
    // Get compute stream or nullptr if stream manager not set
    cudaStream_t GetComputeStream() const;
};

} // namespace beam_search
} // namespace whisper