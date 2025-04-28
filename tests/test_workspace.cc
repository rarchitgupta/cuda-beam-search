#include "whisper/beam_search/beam_types.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace whisper::beam_search;

int main() {
    try {
        // Create workspace with 10MB initial size
        BeamSearchWorkspace workspace(10 * 1024 * 1024);
        std::cout << "Created workspace with capacity: " << 
            (workspace.GetCapacity() / (1024 * 1024)) << " MB" << std::endl;
        
        // Allocate memory for tokens on device
        const size_t num_tokens = 1000000;
        Token* d_tokens = static_cast<Token*>(workspace.Allocate(num_tokens * sizeof(Token)));
        
        // Verify allocation succeeded
        if (!d_tokens) {
            throw std::runtime_error("Failed to allocate memory for tokens");
        }
        
        std::cout << "Allocated memory for " << num_tokens << " tokens" << std::endl;
        std::cout << "Used memory: " << (workspace.GetUsedSize() / (1024 * 1024)) << " MB" << std::endl;
        
        // Create some test tokens on host
        std::vector<Token> h_tokens(10);
        for (int i = 0; i < 10; i++) {
            h_tokens[i] = Token(-0.5f * i, i + 100, i > 0 ? i - 1 : -1);
        }
        
        // Copy test tokens to device
        cudaError_t err = cudaMemcpy(d_tokens, h_tokens.data(), 
                                     10 * sizeof(Token), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy tokens to device: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Copy back and verify
        std::vector<Token> h_result(10);
        err = cudaMemcpy(h_result.data(), d_tokens, 
                        10 * sizeof(Token), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy tokens from device: " + 
                                   std::string(cudaGetErrorString(err)));
        }
        
        // Verify data integrity
        bool passed = true;
        for (int i = 0; i < 10; i++) {
            if (h_result[i].score != h_tokens[i].score ||
                h_result[i].token_id != h_tokens[i].token_id ||
                h_result[i].prev_index != h_tokens[i].prev_index) {
                passed = false;
                std::cout << "Mismatch at index " << i << std::endl;
            }
        }
        
        // Reset workspace
        workspace.Reset();
        std::cout << "Reset workspace, used memory: " << workspace.GetUsedSize() << " bytes" << std::endl;
        
        if (passed) {
            std::cout << "Integration test passed!" << std::endl;
            return 0;
        } else {
            std::cout << "Integration test failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 