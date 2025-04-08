#pragma once

#include <cstdint>
#include <vector>

namespace whisper {
namespace decoder {

// Structure to hold beam state
struct BeamState {
    int32_t* tokens;        // Current token sequence
    float* scores;          // Beam scores
    int32_t* lengths;       // Sequence lengths
    int32_t num_beams;      // Number of active beams
    int32_t max_length;     // Maximum sequence length
};

// Configuration for beam search
struct BeamSearchConfig {
    int32_t num_beams = 5;          // Number of beams to maintain
    int32_t max_length = 448;       // Maximum sequence length
    float length_penalty = 1.0f;    // Length penalty factor
    bool early_stopping = true;     // Whether to stop when all beams reach EOS
};

class BeamSearch {
public:
    BeamSearch(const BeamSearchConfig& config);
    ~BeamSearch();

    // Initialize beam search state
    void Initialize();

    // Process logits for one step
    void ProcessStep(const float* logits, int32_t batch_size, int32_t vocab_size);

    // Get the best token sequences
    void GetResults(std::vector<std::vector<int32_t>>& tokens);

private:
    BeamState beam_state_;
    BeamSearchConfig config_;
    void* device_workspace_;  // CUDA workspace memory
};

} // namespace decoder
} // namespace whisper 