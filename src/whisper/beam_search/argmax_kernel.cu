#include "whisper/beam_search/argmax_kernel.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <cstddef>

namespace whisper {
namespace beam_search {

// Kernel to compute index of max element in data[0..size)
// Launch with one block; only thread 0 does the work.
__global__ void argmax_kernel(const float* data, std::size_t size, int* result) {
    if (threadIdx.x == 0) {
        // Perform serial scan to find max index
        float max_val = (size > 0 ? data[0] : -FLT_MAX);
        int max_i = (size > 0 ? 0 : -1);
        for (std::size_t i = 1; i < size; ++i) {
            float v = data[i];
            if (v > max_val) {
                max_val = v;
                max_i = static_cast<int>(i);
            }
        }
        *result = max_i;
    }
}

} // namespace beam_search
} // namespace whisper 