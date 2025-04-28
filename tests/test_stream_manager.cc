#include "whisper/utils/cuda_stream_manager.h"
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>

using namespace whisper::utils;

// Test basic stream creation and synchronization
void testBasicStreamFunctionality() {
    CudaStreamManager manager;
    
    // Verify streams were created successfully
    cudaStream_t compute_stream = manager.GetComputeStream();
    cudaStream_t transfer_stream = manager.GetTransferStream();
    
    assert(compute_stream != nullptr);
    assert(transfer_stream != nullptr);
    
    // Test synchronization
    manager.Synchronize();
    
    std::cout << "Basic stream functionality test passed!" << std::endl;
}

// Test error checking
void testErrorChecking() {
    CudaStreamManager manager;
    
    // Test successful operation
    bool success = manager.CheckLastError("Test operation");
    assert(success);
    
    std::cout << "Error checking test passed!" << std::endl;
}

// Test with actual CUDA operations
void testWithCudaOperations() {
    CudaStreamManager manager;
    cudaStream_t compute_stream = manager.GetComputeStream();
    
    // Allocate device memory
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)), 
              "Memory allocation failed");
    
    // Initialize with zeros
    CUDA_CHECK(cudaMemsetAsync(d_data, 0, 1024 * sizeof(float), compute_stream),
              "Memset operation failed");
    
    // Synchronize and verify no errors
    manager.Synchronize();
    assert(manager.CheckLastError("CUDA operations"));
    
    // Clean up
    CUDA_CHECK(cudaFree(d_data), "Memory free failed");
    
    std::cout << "CUDA operations test passed!" << std::endl;
}

int main() {
    try {
        testBasicStreamFunctionality();
        testErrorChecking();
        testWithCudaOperations();
        
        std::cout << "All CudaStreamManager tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 