#include "whisper/utils/cuda_stream_manager.h"

namespace whisper {
namespace utils {

CudaStreamManager::CudaStreamManager() {
    // Create compute stream
    CUDA_CHECK(cudaStreamCreate(&compute_stream_), 
               "Failed to create compute stream");
    
    // Create transfer stream
    CUDA_CHECK(cudaStreamCreate(&transfer_stream_), 
               "Failed to create transfer stream");
    
    // Create events for synchronization
    CUDA_CHECK(cudaEventCreate(&compute_event_),
               "Failed to create compute event");
    
    CUDA_CHECK(cudaEventCreate(&transfer_event_),
               "Failed to create transfer event");
}

CudaStreamManager::~CudaStreamManager() {
    // Synchronize and destroy compute stream
    if (compute_stream_) {
        cudaStreamSynchronize(compute_stream_);
        cudaStreamDestroy(compute_stream_);
    }
    
    // Synchronize and destroy transfer stream
    if (transfer_stream_) {
        cudaStreamSynchronize(transfer_stream_);
        cudaStreamDestroy(transfer_stream_);
    }
    
    // Destroy events
    if (compute_event_) {
        cudaEventDestroy(compute_event_);
    }
    
    if (transfer_event_) {
        cudaEventDestroy(transfer_event_);
    }
}

cudaStream_t CudaStreamManager::GetComputeStream() const {
    return compute_stream_;
}

cudaStream_t CudaStreamManager::GetTransferStream() const {
    return transfer_stream_;
}

void CudaStreamManager::Synchronize() {
    // Synchronize both streams
    SynchronizeCompute();
    SynchronizeTransfer();
}

void CudaStreamManager::SynchronizeCompute() {
    if (compute_stream_) {
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_), 
                  "Failed to synchronize compute stream");
    }
}

void CudaStreamManager::SynchronizeTransfer() {
    if (transfer_stream_) {
        CUDA_CHECK(cudaStreamSynchronize(transfer_stream_), 
                  "Failed to synchronize transfer stream");
    }
}

void CudaStreamManager::RecordComputeEvent() {
    CUDA_CHECK(cudaEventRecord(compute_event_, compute_stream_),
              "Failed to record compute event");
}

void CudaStreamManager::RecordTransferEvent() {
    CUDA_CHECK(cudaEventRecord(transfer_event_, transfer_stream_),
              "Failed to record transfer event");
}

void CudaStreamManager::WaitForComputeOnTransfer() {
    CUDA_CHECK(cudaStreamWaitEvent(transfer_stream_, compute_event_, 0),
              "Failed to make transfer stream wait for compute event");
}

void CudaStreamManager::WaitForTransferOnCompute() {
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, transfer_event_, 0),
              "Failed to make compute stream wait for transfer event");
}

bool CudaStreamManager::CheckLastError(const char* operation) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err = std::string(operation) + ": " + 
                          cudaGetErrorString(error);
        throw std::runtime_error(err);
        return false;
    }
    return true;
}

} // namespace utils
} // namespace whisper 