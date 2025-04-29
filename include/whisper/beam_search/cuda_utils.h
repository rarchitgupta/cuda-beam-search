#pragma once
#include <cuda_runtime.h>
#include <cassert>

// Check a CUDA API call returns success
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = (call); \
    assert(err == cudaSuccess && "CUDA error: " #call); \
  } while (0)

// Launch a kernel and check for errors
#define LAUNCH_AND_CHECK(...) do { __VA_ARGS__; cudaError_t err = cudaGetLastError(); assert(err == cudaSuccess && "Kernel launch failed: " #__VA_ARGS__); CUDA_CHECK(cudaDeviceSynchronize()); } while (0)