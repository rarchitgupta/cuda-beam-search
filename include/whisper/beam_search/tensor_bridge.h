// tensor_bridge.h - Interface for passing PyTorch tensors to C++/CUDA
#pragma once

#include <cstdint>
#include <vector>

namespace whisper {
namespace beam_search {

/**
 * TensorBridge class handles transfer of logits tensors from Python to C++
 * Provides utilities for tensor shape management and data transfer
 */
class TensorBridge {
public:
    TensorBridge();
    ~TensorBridge();

    /**
     * Sets the logits tensor from a PyTorch tensor in device memory
     * 
     * @param data Pointer to the tensor data in device memory
     * @param batch_size Number of items in batch
     * @param seq_len Sequence length
     * @param vocab_size Vocabulary size (number of logits per token)
     * @return true if successful
     */
    bool set_logits(float* data, int batch_size, int seq_len, int vocab_size);

    /**
     * Access the tensor data in device memory
     * 
     * @return Pointer to the tensor data in device memory
     */
    float* get_device_data() const;

    /**
     * Get the shape of the tensor
     * 
     * @return Vector containing the dimensions [batch_size, seq_len, vocab_size]
     */
    std::vector<int> get_shape() const;

private:
    float* device_data_;   // Pointer to data in device memory
    int batch_size_;       // Batch size dimension
    int seq_len_;          // Sequence length dimension
    int vocab_size_;       // Vocabulary size dimension
    bool owns_memory_;     // Whether this class owns the device memory
};

} // namespace beam_search
} // namespace whisper 