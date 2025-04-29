#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace whisper {
namespace beam_search {

// GPU kernel to compute the index of the maximum value in a 1D float array.
__global__ void argmax_kernel(
    const float* data,
    std::size_t size,
    int* result
);

}  // namespace beam_search
}  // namespace whisper 