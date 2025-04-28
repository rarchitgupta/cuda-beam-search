#pragma once

#include "whisper/utils/cuda_common.h"
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace whisper {
namespace utils {

/**
 * Manages CUDA streams for compute and memory transfer operations.
 * 
 * Uses two separate streams:
 * 1. Compute stream: for kernel execution
 * 2. Transfer stream: for non-blocking device-to-host transfers
 * 
 * Also provides events for fine-grained synchronization between streams.
 */
class CudaStreamManager {
public:
    // Create streams with default flags
    CudaStreamManager();
    
    // Synchronize and destroy streams
    ~CudaStreamManager();
    
    // Get compute stream for kernel execution
    cudaStream_t GetComputeStream() const;
    
    // Get transfer stream for memory operations
    cudaStream_t GetTransferStream() const;
    
    // Synchronize both streams
    void Synchronize();
    
    // Synchronize only the compute stream
    void SynchronizeCompute();
    
    // Synchronize only the transfer stream
    void SynchronizeTransfer();
    
    // Record an event on the compute stream
    void RecordComputeEvent();
    
    // Record an event on the transfer stream
    void RecordTransferEvent();
    
    // Make transfer stream wait for compute stream to finish
    void WaitForComputeOnTransfer();
    
    // Make compute stream wait for transfer stream to finish
    void WaitForTransferOnCompute();
    
    // Check last error and throw exception if non-success
    bool CheckLastError(const char* operation);

private:
    cudaStream_t compute_stream_;
    cudaStream_t transfer_stream_;
    cudaEvent_t compute_event_;
    cudaEvent_t transfer_event_;
};

} // namespace utils
} // namespace whisper 