#pragma once

#include <cstddef>
#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_array.h"

namespace whisper {
namespace beam_search {

// Processes logits from Whisper model and applies sampling strategies (temperature, topK, topP)
class LogitProcessor {
public:
    LogitProcessor(BeamSearchWorkspace* workspace,
                  float temperature = 1.0f,
                  int topK = 0,
                  float topP = 1.0f);

    bool processLogits(const float* deviceLogits,
                       int batchSize,
                       int seqLen,
                       int vocabSize);

    void scoreNextTokens(const BeamArray* beam,
                         int batchIndex,
                         int position,
                         BeamArray* outputBeam);

    void scoreAndPrune(const BeamArray* beam,
                       int batchIndex,
                       int position,
                       BeamArray* outputBeam,
                       std::size_t beamWidth);

    void setSamplingParams(float temperature,
                           int topK = 0,
                           float topP = 1.0f);

private:
    float* processedLogits_ = nullptr;
    int* tokenIndices_ = nullptr;
    float* tempStorage_ = nullptr;
    
    float temperature_ = 1.0f;
    int topK_ = 0;
    float topP_ = 1.0f;
    
    int batchSize_ = 0;
    int seqLen_ = 0;
    int vocabSize_ = 0;
    
    BeamSearchWorkspace* workspace_ = nullptr;
    std::size_t tempStorageSize_ = 0;
    
    void allocateMemory(std::size_t requiredSize);
    
    void applyTemperature(float* deviceLogits,
                          int batchIndex,
                          int position);
    
    void applySoftmax(float* deviceLogits,
                      int batchIndex,
                      int position);
    
    void applyTopK(int batchIndex,
                   int position);
    
    void applyTopP(int batchIndex,
                   int position);
};

} // namespace beam_search
} // namespace whisper