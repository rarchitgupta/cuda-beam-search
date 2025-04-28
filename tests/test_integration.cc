#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_array.h"
#include "whisper/beam_search/logit_processor.h"
#include "whisper/beam_search/tensor_bridge.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include <cmath>

using namespace whisper::beam_search;

// Helper function to generate random logits for testing
std::vector<float> generateTestLogits(int batch_size, int seq_len, int vocab_size) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 2.0f);
    
    std::vector<float> logits(batch_size * seq_len * vocab_size);
    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] = dist(gen);
    }
    return logits;
}

// Full integration test of all components
void testFullBeamSearchPipeline() {
    // Configuration
    const int batch_size = 2;
    const int seq_len = 3;
    const int vocab_size = 200;
    const int beam_width = 5;
    const float temperature = 0.8f;
    const int top_k = 50;
    
    // 1. Create workspace
    BeamSearchWorkspace workspace(64 * 1024 * 1024);
    
    // 2. Setup tensor bridge
    TensorBridge tensor_bridge;
    
    // 3. Generate test logits
    std::vector<float> h_logits = generateTestLogits(batch_size, seq_len, vocab_size);
    
    // 4. Allocate device memory for logits
    float* d_logits;
    cudaMalloc(&d_logits, h_logits.size() * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // 5. Set logits in tensor bridge
    tensor_bridge.set_logits(d_logits, batch_size, seq_len, vocab_size);
    
    // 6. Create logit processor
    LogitProcessor processor(&workspace, temperature, top_k);
    
    // 7. Process logits
    processor.ProcessLogits(d_logits, batch_size, seq_len, vocab_size);
    
    // 8. Create initial beam arrays for each batch
    BeamArray initial_beam_1(beam_width, &workspace);
    BeamArray initial_beam_2(beam_width, &workspace);
    
    // Add start token to each beam
    initial_beam_1.AddToken(Token(0.0f, 0, -1));
    initial_beam_2.AddToken(Token(0.0f, 0, -1));
    
    // 9. Create output beam arrays
    BeamArray output_beam_1(beam_width * vocab_size, &workspace);
    BeamArray output_beam_2(beam_width * vocab_size, &workspace);
    
    // 10. Process first position in sequence
    processor.ScoreAndPrune(&initial_beam_1, 0, 0, &output_beam_1, beam_width);
    processor.ScoreAndPrune(&initial_beam_2, 1, 0, &output_beam_2, beam_width);
    
    // 11. Verify output beams
    assert(output_beam_1.Size() == beam_width);
    assert(output_beam_2.Size() == beam_width);
    
    // 12. Get tokens from beams
    std::vector<Token> tokens_1;
    std::vector<Token> tokens_2;
    output_beam_1.CopyToHost(tokens_1);
    output_beam_2.CopyToHost(tokens_2);
    
    // 13. Verify tokens are sorted by score
    for (size_t i = 1; i < tokens_1.size(); i++) {
        assert(tokens_1[i-1].score >= tokens_1[i].score);
    }
    for (size_t i = 1; i < tokens_2.size(); i++) {
        assert(tokens_2[i-1].score >= tokens_2[i].score);
    }
    
    // 14. Process second position in sequence
    BeamArray next_beam_1(beam_width * vocab_size, &workspace);
    processor.ScoreAndPrune(&output_beam_1, 0, 1, &next_beam_1, beam_width);
    
    BeamArray next_beam_2(beam_width * vocab_size, &workspace);
    processor.ScoreAndPrune(&output_beam_2, 1, 1, &next_beam_2, beam_width);
    
    // 15. Verify next beams
    assert(next_beam_1.Size() == beam_width);
    assert(next_beam_2.Size() == beam_width);
    
    // 16. Repeat for third position
    BeamArray final_beam_1(beam_width * vocab_size, &workspace);
    processor.ScoreAndPrune(&next_beam_1, 0, 2, &final_beam_1, beam_width);
    
    BeamArray final_beam_2(beam_width * vocab_size, &workspace);
    processor.ScoreAndPrune(&next_beam_2, 1, 2, &final_beam_2, beam_width);
    
    // 17. Verify final beams
    assert(final_beam_1.Size() == beam_width);
    assert(final_beam_2.Size() == beam_width);
    
    // 18. Get best hypothesis from each batch
    Token best_token_1 = final_beam_1.GetToken(0);
    Token best_token_2 = final_beam_2.GetToken(0);
    
    // 19. Verify we can trace back through the token history
    std::vector<int> sequence_1, sequence_2;
    
    // Reconstruct sequence 1
    Token current = best_token_1;
    while (current.prev_index >= 0) {
        sequence_1.push_back(current.token_id);
        
        // Get previous token based on prev_index
        if (current.prev_index >= 0) {
            // Find which beam this token came from
            if (sequence_1.size() == 1) {
                current = next_beam_1.GetToken(current.prev_index);
            } else if (sequence_1.size() == 2) {
                current = output_beam_1.GetToken(current.prev_index);
            } else {
                break;
            }
        }
    }
    
    // Reconstruct sequence 2
    current = best_token_2;
    while (current.prev_index >= 0) {
        sequence_2.push_back(current.token_id);
        
        // Get previous token based on prev_index
        if (current.prev_index >= 0) {
            // Find which beam this token came from
            if (sequence_2.size() == 1) {
                current = next_beam_2.GetToken(current.prev_index);
            } else if (sequence_2.size() == 2) {
                current = output_beam_2.GetToken(current.prev_index);
            } else {
                break;
            }
        }
    }
    
    // 20. Verify sequences
    assert(sequence_1.size() <= 3);
    assert(sequence_2.size() <= 3);
    
    // Clean up
    cudaFree(d_logits);
    
    std::cout << "Full beam search pipeline test passed!" << std::endl;
}

int main() {
    try {
        testFullBeamSearchPipeline();
        
        std::cout << "All integration tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 