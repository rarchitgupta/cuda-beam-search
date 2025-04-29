#include "whisper/beam_search/argmax_kernel.h"
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>

using namespace whisper::beam_search;

int main() {
    // Test parameters
    const int length = 100000;

    // Generate random data on host
    std::vector<float> h_data(length);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < length; ++i) {
        h_data[i] = dist(gen);
    }

    // Compute expected argmax on host
    auto it = std::max_element(h_data.begin(), h_data.end());
    int expected_idx = static_cast<int>(std::distance(h_data.begin(), it));

    // Allocate device memory
    float* d_data = nullptr;
    int* d_result = nullptr;
    cudaError_t err = cudaMalloc(&d_data, length * sizeof(float));
    assert(err == cudaSuccess);
    err = cudaMalloc(&d_result, sizeof(int));
    assert(err == cudaSuccess);

    // Copy data to device
    err = cudaMemcpy(d_data, h_data.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // Host debug: launching kernel
    std::cout << "[DEBUG] Launching argmax_kernel<<<1,1>>>()" << std::endl;
    // Clear any existing CUDA error
    cudaGetLastError();
    // Launch argmax kernel with one block and 1 thread (serial scan)
    argmax_kernel<<<1, 1>>>(d_data, length, d_result);
    // Check for launch errors
    cudaError_t errLaunch = cudaGetLastError();
    std::cout << "[DEBUG] cudaGetLastError after launch: " << cudaGetErrorString(errLaunch) << std::endl;
    err = cudaDeviceSynchronize();
    std::cout << "[DEBUG] cudaDeviceSynchronize returned: " << cudaGetErrorString(err) << std::endl;
    assert(err == cudaSuccess);

    // Copy result back
    int h_idx = -1;
    err = cudaMemcpy(&h_idx, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    // Debug: print expected vs actual GPU result
    std::cout << "[DEBUG] expected idx = " << expected_idx 
              << ", GPU idx = " << h_idx << std::endl;

    // Validate result
    assert(h_idx == expected_idx);

    std::cout << "Argmax kernel test passed! idx=" << h_idx << std::endl;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_result);
    return 0;
} 