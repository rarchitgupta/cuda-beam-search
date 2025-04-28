#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>
#include "whisper/utils/cuda_stream_manager.h"

using namespace whisper::utils;

// Simple non-kernel function to test stream synchronization
void testEventSynchronization() {
    CudaStreamManager stream_manager;
    
    // Allocate device memory
    int *d_compute_result, *d_transfer_result;
    cudaMalloc(&d_compute_result, sizeof(int));
    cudaMalloc(&d_transfer_result, sizeof(int));
    
    // Initialize device memory
    cudaMemset(d_compute_result, 0, sizeof(int));
    cudaMemset(d_transfer_result, 0, sizeof(int));
    
    // Set up timing measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Execute operations on compute stream
    // Use cudaMemsetAsync as a simple operation that runs on the stream
    cudaMemsetAsync(d_compute_result, 42, sizeof(int), stream_manager.GetComputeStream());
    
    // Add a small sleep to simulate work
    cudaStreamSynchronize(stream_manager.GetComputeStream());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Record event on compute stream
    stream_manager.RecordComputeEvent();
    
    // Execute operation on transfer stream
    cudaMemsetAsync(d_transfer_result, 100, sizeof(int), stream_manager.GetTransferStream());
    
    // Add a small sleep to simulate work
    cudaStreamSynchronize(stream_manager.GetTransferStream());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Now make transfer stream wait for compute stream to finish
    stream_manager.WaitForComputeOnTransfer();
    
    // Execute another operation on transfer stream
    cudaMemsetAsync(d_transfer_result, 50, sizeof(int), stream_manager.GetTransferStream());
    
    // Add a small sleep to simulate work
    cudaStreamSynchronize(stream_manager.GetTransferStream());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Synchronize both streams
    stream_manager.Synchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Read back results
    int h_compute_result, h_transfer_result;
    cudaMemcpy(&h_compute_result, d_compute_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_transfer_result, d_transfer_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify results
    assert(h_compute_result == 42);
    assert(h_transfer_result == 50);
    
    // Free device memory
    cudaFree(d_compute_result);
    cudaFree(d_transfer_result);
    
    // With the sleeps, this should be approximately 350ms
    // But we'll just make sure it's a reasonable value for now
    std::cout << "Total execution time: " << elapsed << "ms" << std::endl;
    assert(elapsed > 300 && elapsed < 400);
    
    std::cout << "Event synchronization test passed!" << std::endl;
}

void testSeparateStreamSynchronization() {
    CudaStreamManager stream_manager;
    
    // Allocate and initialize device memory
    int *d_compute_data, *d_transfer_data;
    cudaMalloc(&d_compute_data, sizeof(int));
    cudaMalloc(&d_transfer_data, sizeof(int));
    cudaMemset(d_compute_data, 0, sizeof(int));
    cudaMemset(d_transfer_data, 0, sizeof(int));
    
    // Launch operations on both streams
    cudaMemsetAsync(d_compute_data, 100, sizeof(int), stream_manager.GetComputeStream());
    cudaMemsetAsync(d_transfer_data, 150, sizeof(int), stream_manager.GetTransferStream());
    
    // Simulate work with sleeps
    cudaStreamSynchronize(stream_manager.GetComputeStream());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Synchronize only the compute stream
    stream_manager.SynchronizeCompute();
    
    // At this point, the compute stream should be done, but transfer stream might still be running
    int compute_result;
    cudaMemcpy(&compute_result, d_compute_data, sizeof(int), cudaMemcpyDeviceToHost);
    assert(compute_result == 100); // Should have the value set by the compute kernel
    
    // Simulate more work on transfer stream
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Now synchronize the transfer stream
    stream_manager.SynchronizeTransfer();
    
    // At this point, both streams should be done
    int transfer_result;
    cudaMemcpy(&transfer_result, d_transfer_data, sizeof(int), cudaMemcpyDeviceToHost);
    assert(transfer_result == 150); // Should have the value set by the transfer kernel
    
    // Clean up
    cudaFree(d_compute_data);
    cudaFree(d_transfer_data);
    
    std::cout << "Separate stream synchronization test passed!" << std::endl;
}

int main() {
    try {
        testEventSynchronization();
        testSeparateStreamSynchronization();
        
        std::cout << "All stream synchronization tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
} 