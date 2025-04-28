#pragma once

#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_array.h"
#include <cuda_runtime.h>
#include <vector>

namespace whisper {
namespace beam_search {

// Processes logits from Whisper model and applies sampling strategies (temperature, top-k, top-p)
class LogitProcessor {
public:
    LogitProcessor(BeamSearchWorkspace* workspace, float temperature = 1.0f, 
                  int top_k = 0, float top_p = 1.0f);

    bool ProcessLogits(const float* d_logits, int batch_size, int seq_len, int vocab_size);

    void ScoreNextTokens(const BeamArray* beam, int batch_index, int position, BeamArray* output_beam);

    // Convenience method to score tokens and prune to beam_width in one step
    void ScoreAndPrune(const BeamArray* beam, int batch_index, int position, 
                      BeamArray* output_beam, size_t beam_width);

    void SetSamplingParams(float temperature, int top_k = 0, float top_p = 1.0f);

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
    
    void AllocateMemory(size_t required_size);
    
    void ApplyTemperature(float* d_logits, int batch_index, int position);
    
    void ApplySoftmax(float* d_logits, int batch_index, int position);
    
    void ApplyTopK(int batch_index, int position);
    
    void ApplyTopP(int batch_index, int position);
};

} // namespace beam_search
} // namespace whisper