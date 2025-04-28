#pragma once

#include <cstdint>
#include <vector>
#include "whisper/beam_search/beam_config.h"

namespace whisper {
namespace decoder {

// State maintained during beam search decoding
struct BeamState {
    int32_t* tokens;     // Current token sequences
    float* scores;       // Beam scores
    int32_t* lengths;
    int32_t num_beams;
    int32_t max_length;
};

// Main decoder that implements beam search on GPU
class BeamSearch {
public:
    BeamSearch(const beam_search::BeamSearchConfig& config);
    ~BeamSearch();

    void Initialize();

    void ProcessStep(const float* logits, int32_t batch_size, int32_t vocab_size);

    void GetResults(std::vector<std::vector<int32_t>>& tokens);

private:
    BeamState beam_state_;
    beam_search::BeamSearchConfig config_;
    void* device_workspace_;
};

} // namespace decoder
} // namespace whisper