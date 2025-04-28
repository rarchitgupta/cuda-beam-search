#include "whisper/beam_search/tensor_bridge.h"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace whisper::beam_search;

// Helper class for half-precision
struct half_t {
    unsigned short bits;
};

class TensorBridgeTest {
public:
    void SetUp() {
        // Nothing to do
    }

    void TearDown() {
        // Clean up any allocated device memory
        if (d_float_data_) {
            cudaFree(d_float_data_);
            d_float_data_ = nullptr;
        }
        if (d_half_data_) {
            cudaFree(d_half_data_);
            d_half_data_ = nullptr;
        }
    }

    // Allocate and initialize device memory for testing
    void AllocateFloatTensor(int batch_size, int seq_len, int vocab_size) {
        size_t size = batch_size * seq_len * vocab_size * sizeof(float);
        cudaMalloc(&d_float_data_, size);
        batch_size_ = batch_size;
        seq_len_ = seq_len;
        vocab_size_ = vocab_size;
    }

    // Allocate and initialize device memory for half-precision testing
    void AllocateHalfTensor(int batch_size, int seq_len, int vocab_size) {
        size_t size = batch_size * seq_len * vocab_size * sizeof(half_t);
        cudaMalloc(&d_half_data_, size);
        batch_size_ = batch_size;
        seq_len_ = seq_len;
        vocab_size_ = vocab_size;
    }

    void RunTests() {
        TestFloatTensor();
        TestHalfTensor();
    }

private:
    void TestFloatTensor() {
        // Allocate a test tensor
        AllocateFloatTensor(2, 16, 128);
        
        // Create TensorBridge
        TensorBridge bridge;
        
        // Set logits
        bool result = bridge.set_logits(d_float_data_, batch_size_, seq_len_, vocab_size_);
        assert(result == true);
        
        // Check shape
        std::vector<int> shape = bridge.get_shape();
        assert(shape.size() == 3);
        assert(shape[0] == batch_size_);
        assert(shape[1] == seq_len_);
        assert(shape[2] == vocab_size_);
        
        // Check data pointer
        assert(bridge.get_device_data() == d_float_data_);
        
        // Check dtype
        assert(bridge.get_dtype() == TensorDType::FLOAT32);
        
        std::cout << "Float tensor tests passed!" << std::endl;
    }

    void TestHalfTensor() {
        // Allocate a test tensor
        AllocateHalfTensor(2, 16, 128);
        
        // Create TensorBridge
        TensorBridge bridge;
        
        // Set logits with half precision
        bool result = bridge.set_logits_half(d_half_data_, batch_size_, seq_len_, vocab_size_);
        assert(result == true);
        
        // Check shape
        std::vector<int> shape = bridge.get_shape();
        assert(shape.size() == 3);
        assert(shape[0] == batch_size_);
        assert(shape[1] == seq_len_);
        assert(shape[2] == vocab_size_);
        
        // Check data pointer
        assert(bridge.get_device_data() == d_half_data_);
        
        // Check dtype
        assert(bridge.get_dtype() == TensorDType::FLOAT16);
        
        std::cout << "Half tensor tests passed!" << std::endl;
    }

    float* d_float_data_ = nullptr;
    void* d_half_data_ = nullptr;
    int batch_size_ = 0;
    int seq_len_ = 0;
    int vocab_size_ = 0;
};

int main() {
    try {
        TensorBridgeTest test;
        test.SetUp();
        test.RunTests();
        test.TearDown();
        
        std::cout << "All TensorBridge tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 