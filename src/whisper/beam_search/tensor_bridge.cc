#include "whisper/beam_search/tensor_bridge.h"
#include "whisper/beam_search/cuda_utils.h"  // For CUDA_CHECK
#include <cuda_runtime.h>
#include <stdexcept>

namespace whisper {
namespace beam_search {

TensorBridge::TensorBridge()
    : deviceData_(nullptr), batchSize_(0), seqLen_(0), vocabSize_(0), ownsMemory_(false) {}

TensorBridge::~TensorBridge() {
    // Free device memory if this object owns it
    if (ownsMemory_ && deviceData_ != nullptr) {
        CUDA_CHECK(cudaFree(deviceData_));
        deviceData_ = nullptr;
    }
}

bool TensorBridge::setLogits(float* deviceData,
                            std::size_t batchSize,
                            std::size_t seqLen,
                            std::size_t vocabSize) {
    deviceData_ = deviceData;
    batchSize_ = batchSize;
    seqLen_ = seqLen;
    vocabSize_ = vocabSize;
    ownsMemory_ = false;  // We don't own this memory - framework does
    return true;
}

float* TensorBridge::getDeviceData() const {
    return deviceData_;
}

std::vector<std::size_t> TensorBridge::getShape() const {
    return {batchSize_, seqLen_, vocabSize_};
}

} // namespace beam_search
} // namespace whisper 