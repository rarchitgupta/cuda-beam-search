#pragma once

#include "whisper/utils/cuda_common.h"
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <cstddef>

namespace whisper {
namespace utils {

/**
 * GPU memory manager that provides a unified interface for memory allocation
 * from a pre-allocated pool, avoiding frequent cudaMalloc/cudaFree calls.
 */
class GPUMemoryPool {
public:
    /**
     * Constructor with optional initial size.
     * 
     * @param initial_size Initial pool size in bytes (default: 16MB)
     */
    GPUMemoryPool(size_t initial_size = 16 * 1024 * 1024);
    
    /**
     * Destructor - frees all allocated GPU memory.
     */
    ~GPUMemoryPool();
    
    /**
     * Allocate memory from the pool.
     * 
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment in bytes (default: 256 bytes)
     * @return Pointer to allocated GPU memory
     */
    void* Allocate(size_t size, size_t alignment = 256);
    
    /**
     * Free a specific pointer (not implemented in pooled version)
     * 
     * @param ptr Pointer to free (ignored in pooled version)
     */
    void Free(void* ptr);
    
    /**
     * Reset the pool to empty state without freeing GPU memory.
     * All previous allocations become invalid after this call.
     */
    void Reset();
    
    /**
     * Ensure the pool has at least the specified capacity.
     * 
     * @param size Minimum capacity to ensure
     */
    void EnsureCapacity(size_t size);
    
    /**
     * Get the current amount of used memory in bytes.
     * 
     * @return Used memory in bytes
     */
    size_t GetUsedSize() const;
    
    /**
     * Get the total capacity of the memory pool in bytes.
     * 
     * @return Total capacity in bytes
     */
    size_t GetCapacity() const;
    
private:
    void* d_memory_;       // Device memory pointer
    size_t capacity_;      // Total capacity in bytes
    size_t used_;          // Used bytes
    
    // Disallow copying
    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;
};

} // namespace utils
} // namespace whisper 