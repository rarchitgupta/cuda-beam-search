#pragma once

#include <cstdint>
#include <vector>

namespace whisper {
namespace beam_search {

// Interface for passing PyTorch/TensorFlow tensors to CUDA beam search
class TensorBridge {
public:
    TensorBridge();
    ~TensorBridge();

    // Associate logits tensor from ML framework (doesn't take ownership)
    bool set_logits(float* data, int batch_size, int seq_len, int vocab_size);

    float* get_device_data() const;

    std::vector<int> get_shape() const;

private:
    float* device_data_;
    int batch_size_;
    int seq_len_;
    int vocab_size_;
    bool owns_memory_;
};

} // namespace beam_search
} // namespace whisper