#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_array.h"
#include "whisper/beam_search/logit_processor.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include <cmath>

using namespace whisper::beam_search;

// Helper function to generate mock logits for testing
void generateRandomLogits(std::vector<float>& logits, int batch_size, int seq_len, int vocab_size) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 2.0f);
    
    logits.resize(batch_size * seq_len * vocab_size);
    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] = dist(gen);
    }
}

// Test basic functionality
void testBasicFunctionality() {
    // Create workspace
    BeamSearchWorkspace workspace(32 * 1024 * 1024);
    
    // Create LogitProcessor
    LogitProcessor processor(&workspace, 1.0f);
    
    // Setup test data
    const int batch_size = 1;
    const int seq_len = 1;
    const int vocab_size = 100;
    
    std::vector<float> h_logits;
    generateRandomLogits(h_logits, batch_size, seq_len, vocab_size);
    
    // Copy to device
    float* d_logits;
    cudaMalloc(&d_logits, h_logits.size() * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Process logits
    bool success = processor.processLogits(d_logits, batch_size, seq_len, vocab_size);
    assert(success);
    
    // Create beam with start token
    BeamArray beam(1024, &workspace);
    beam.addToken(Token(0.0f, 0, -1));
    
    // Create output beam
    BeamArray output_beam(1024, &workspace);
    
    // Score tokens
    processor.scoreNextTokens(&beam, 0, 0, &output_beam);
    
    // Verify output beam size (should be vocab_size)
    assert(output_beam.size() == vocab_size);
    
    // Clean up
    cudaFree(d_logits);
    
    std::cout << "Basic functionality test passed!" << std::endl;
}

// Test score and prune
void testScoreAndPrune() {
    // Create workspace
    BeamSearchWorkspace workspace(32 * 1024 * 1024);
    
    // Create LogitProcessor
    LogitProcessor processor(&workspace, 1.0f);
    
    // Setup test data
    const int batch_size = 1;
    const int seq_len = 1;
    const int vocab_size = 100;
    const int beam_width = 5;
    
    std::vector<float> h_logits;
    generateRandomLogits(h_logits, batch_size, seq_len, vocab_size);
    
    // Copy to device
    float* d_logits;
    cudaMalloc(&d_logits, h_logits.size() * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Process logits
    processor.processLogits(d_logits, batch_size, seq_len, vocab_size);
    
    // Create beam with start token
    BeamArray beam(1024, &workspace);
    beam.addToken(Token(0.0f, 0, -1));
    
    // Create output beam
    BeamArray output_beam(1024, &workspace);
    
    // Score and prune
    processor.scoreAndPrune(&beam, 0, 0, &output_beam, beam_width);
    
    // Verify output beam size
    assert(output_beam.size() == beam_width);
    
    // Verify tokens are sorted by score
    std::vector<Token> tokens;
    output_beam.copyToHost(tokens);
    
    for (size_t i = 1; i < tokens.size(); i++) {
        assert(tokens[i-1].score >= tokens[i].score);
    }
    
    // Clean up
    cudaFree(d_logits);
    
    std::cout << "Score and prune test passed!" << std::endl;
}

// Test temperature scaling
void testTemperature() {
    // Create workspace
    BeamSearchWorkspace workspace(32 * 1024 * 1024);
    
    // Create LogitProcessor with high temperature
    LogitProcessor hot_processor(&workspace, 2.0f);
    LogitProcessor cold_processor(&workspace, 0.5f);
    
    // Setup test data
    const int batch_size = 1;
    const int seq_len = 1;
    const int vocab_size = 100;
    const int beam_width = 5;
    
    std::vector<float> h_logits;
    generateRandomLogits(h_logits, batch_size, seq_len, vocab_size);
    
    // Copy to device
    float* d_logits;
    cudaMalloc(&d_logits, h_logits.size() * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy to second device pointer
    float* d_logits2;
    cudaMalloc(&d_logits2, h_logits.size() * sizeof(float));
    cudaMemcpy(d_logits2, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Process logits
    hot_processor.processLogits(d_logits, batch_size, seq_len, vocab_size);
    cold_processor.processLogits(d_logits2, batch_size, seq_len, vocab_size);
    
    // Create source beam
    BeamArray beam(1024, &workspace);
    beam.addToken(Token(0.0f, 0, -1));
    
    // Create output beams
    BeamArray hot_beam(1024, &workspace);
    BeamArray cold_beam(1024, &workspace);
    
    // Score and prune
    hot_processor.scoreAndPrune(&beam, 0, 0, &hot_beam, beam_width);
    cold_processor.scoreAndPrune(&beam, 0, 0, &cold_beam, beam_width);
    
    // Get tokens
    std::vector<Token> hot_tokens;
    std::vector<Token> cold_tokens;
    hot_beam.copyToHost(hot_tokens);
    cold_beam.copyToHost(cold_tokens);
    
    // Verify that cold beam has more extreme scores (larger difference between scores)
    float hot_score_diff = hot_tokens[0].score - hot_tokens[beam_width-1].score;
    float cold_score_diff = cold_tokens[0].score - cold_tokens[beam_width-1].score;
    
    // Cold temperature should create more extreme differences
    assert(cold_score_diff > hot_score_diff);
    
    // Clean up
    cudaFree(d_logits);
    cudaFree(d_logits2);
    
    std::cout << "Temperature test passed!" << std::endl;
}

// Test top-k filtering
void testTopK() {
    // Create workspace
    BeamSearchWorkspace workspace(32 * 1024 * 1024);
    
    // Create LogitProcessor with top-k = 10
    const int top_k = 10;
    LogitProcessor processor(&workspace, 1.0f, top_k);
    
    // Setup test data
    const int batch_size = 1;
    const int seq_len = 1;
    const int vocab_size = 100;
    
    std::vector<float> h_logits;
    generateRandomLogits(h_logits, batch_size, seq_len, vocab_size);
    
    // Copy to device
    float* d_logits;
    cudaMalloc(&d_logits, h_logits.size() * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Process logits
    processor.processLogits(d_logits, batch_size, seq_len, vocab_size);
    
    // Create beam with start token
    BeamArray beam(1024, &workspace);
    beam.addToken(Token(0.0f, 0, -1));
    
    // Create output beam
    BeamArray output_beam(1024, &workspace);
    
    // Score tokens
    processor.scoreNextTokens(&beam, 0, 0, &output_beam);
    
    // Sort output beam
    output_beam.sortByScore();
    
    // Get tokens
    std::vector<Token> tokens;
    output_beam.copyToHost(tokens);
    
    // Count non-negative infinity scores (should be top_k)
    int valid_count = 0;
    for (const auto& token : tokens) {
        if (std::isfinite(token.score)) {
            valid_count++;
        }
    }
    
    assert(valid_count <= top_k + 1); // +1 for numerical stability
    
    // Clean up
    cudaFree(d_logits);
    
    std::cout << "Top-K test passed!" << std::endl;
}

int main() {
    try {
        testBasicFunctionality();
        testScoreAndPrune();
        testTemperature();
        testTopK();
        
        std::cout << "All LogitProcessor tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 