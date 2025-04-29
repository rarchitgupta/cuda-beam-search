#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace whisper {
namespace beam_search {

// Interface for passing framework logits tensors to CUDA beam search.
class TensorBridge {
public:
    // Constructs an empty TensorBridge.
    TensorBridge();

    // Destroys the TensorBridge; does not free tensor memory.
    ~TensorBridge();

    // Sets the logits buffer (raw device pointer) without taking ownership.
    bool setLogits(float* deviceData,
                   std::size_t batchSize,
                   std::size_t seqLen,
                   std::size_t vocabSize);

    // Returns the raw device pointer for the logits.
    float* getDeviceData() const;

    // Returns the shape [batchSize, seqLen, vocabSize].
    std::vector<std::size_t> getShape() const;

private:
    float* deviceData_ = nullptr;
    std::size_t batchSize_ = 0;
    std::size_t seqLen_ = 0;
    std::size_t vocabSize_ = 0;
    bool ownsMemory_ = false;
};

} // namespace beam_search
} // namespace whisper