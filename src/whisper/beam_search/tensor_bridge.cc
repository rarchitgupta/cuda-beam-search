#include "whisper/beam_search/tensor_bridge.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace whisper {
namespace beam_search {

TensorBridge::TensorBridge() 
    : device_data_(nullptr), batch_size_(0), seq_len_(0), vocab_size_(0), owns_memory_(false) {}

TensorBridge::~TensorBridge() {
    // Free device memory if this object owns it
    if (owns_memory_ && device_data_ != nullptr) {
        cudaFree(device_data_);
        device_data_ = nullptr;
    }
}

bool TensorBridge::set_logits(float* data, int batch_size, int seq_len, int vocab_size) {
    device_data_ = data;
    batch_size_ = batch_size;
    seq_len_ = seq_len;
    vocab_size_ = vocab_size;
    owns_memory_ = false;  // We don't own this memory - PyTorch does
    
    return true;
}

float* TensorBridge::get_device_data() const {
    return device_data_;
}

std::vector<int> TensorBridge::get_shape() const {
    return {batch_size_, seq_len_, vocab_size_};
}

} // namespace beam_search
} // namespace whisper 