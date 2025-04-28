#pragma once

#include <cstdint>
#include <vector>

namespace whisper {
namespace beam_search {

// Unified configuration for beam search decoding
struct BeamSearchConfig {
    // Common parameters
    int beam_width = 5;                // Number of beams to track (equivalent to num_beams)
    int max_length = 448;              // Maximum sequence length
    
    // Sampling parameters
    float temperature = 1.0f;          // Temperature for logits
    int top_k = 0;                     // Top-K sampling (0 = disabled)
    float top_p = 1.0f;                // Top-P (nucleus) sampling threshold (1.0 = disabled)
    
    // Specialized parameters
    float length_penalty = 1.0f;        // Higher values favor longer sequences
    bool early_stopping = true;         // Whether to stop when all beams are finished
    std::vector<int> stop_token_ids;    // Token IDs that signify sequence end
    bool use_batch_processing = true;   // Enable batch processing for multiple sequences
    
    // Constructor with minimal required parameters
    BeamSearchConfig(int beam_width = 5, int max_length = 448) 
        : beam_width(beam_width), max_length(max_length) {}
};

} // namespace beam_search
} // namespace whisper 