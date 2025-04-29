#include "whisper/beam_search/beam_array.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

using namespace whisper::beam_search;

void test_beam_array_basic() {
    BeamSearchWorkspace workspace(1024 * 1024);
    BeamArray beam(128, &workspace);
    
    assert(beam.size() == 0);
    assert(beam.capacity() >= 128);
    
    Token t1(1.5f, 42, 10);
    int idx = beam.addToken(t1);
    assert(idx == 0);
    assert(beam.size() == 1);
    
    Token t2 = beam.getToken(0);
    assert(t2.score == 1.5f);
    assert(t2.tokenId == 42);
    assert(t2.prevIndex == 10);
    
    std::cout << "Basic BeamArray tests passed!" << std::endl;
}

void test_beam_array_batch() {
    BeamSearchWorkspace workspace(1024 * 1024);
    BeamArray beam(128, &workspace);
    
    std::vector<Token> tokens;
    for (int i = 0; i < 64; i++) {
        tokens.push_back(Token(static_cast<float>(i), i * 2, i - 1));
    }
    
    int count = beam.addTokens(tokens.data(), tokens.size());
    assert(count == 64);
    assert(beam.size() == 64);
    
    for (int i = 0; i < 64; i++) {
        Token t = beam.getToken(i);
        assert(t.score == static_cast<float>(i));
        assert(t.tokenId == i * 2);
        assert(t.prevIndex == i - 1);
    }
    
    std::cout << "Batch token addition tests passed!" << std::endl;
}

void test_beam_array_sort_prune() {
    BeamSearchWorkspace workspace(1024 * 1024);
    BeamArray beam(128, &workspace);
    
    std::vector<Token> tokens;
    std::vector<float> scores;
    
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    
    for (int i = 0; i < 100; i++) {
        float score = dist(gen);
        scores.push_back(score);
        tokens.push_back(Token(score, i, i - 1));
    }
    
    beam.addTokens(tokens.data(), tokens.size());
    assert(beam.size() == 100);
    
    beam.sortByScore();
    
    Token prev = beam.getToken(0);
    for (int i = 1; i < 100; i++) {
        Token current = beam.getToken(i);
        assert(prev.score >= current.score);
        prev = current;
    }
    
    beam.prune(10);
    assert(beam.size() == 10);
    
    std::sort(scores.begin(), scores.end(), std::greater<float>());
    for (int i = 0; i < 10; i++) {
        Token t = beam.getToken(i);
        assert(t.score == scores[i]);
    }
    
    std::cout << "Sort and prune tests passed!" << std::endl;
}

void test_beam_array_copy_to_host() {
    BeamSearchWorkspace workspace(1024 * 1024);
    BeamArray beam(128, &workspace);
    
    for (int i = 0; i < 50; i++) {
        beam.addToken(Token(static_cast<float>(i), i, i - 1));
    }
    
    std::vector<Token> host_tokens;
    beam.copyToHost(host_tokens);
    
    assert(host_tokens.size() == 50);
    for (int i = 0; i < 50; i++) {
        assert(host_tokens[i].score == static_cast<float>(i));
        assert(host_tokens[i].tokenId == i);
        assert(host_tokens[i].prevIndex == i - 1);
    }
    
    std::cout << "Copy to host tests passed!" << std::endl;
}

void test_beam_array_capacity() {
    BeamSearchWorkspace workspace(1024 * 1024);
    BeamArray beam(10, &workspace);
    
    assert(beam.capacity() >= 10);
    
    std::vector<Token> tokens(20);
    for (int i = 0; i < 20; i++) {
        tokens[i] = Token(static_cast<float>(i), i, i - 1);
    }
    
    int count = beam.addTokens(tokens.data(), tokens.size());
    assert(count == 20);
    assert(beam.size() == 20);
    assert(beam.capacity() >= 20);
    
    std::cout << "Capacity growth tests passed!" << std::endl;
}

void test_beam_array_history_reconstruct() {
    BeamSearchWorkspace workspace(1024 * 1024);
    constexpr int beamWidth = 3;
    constexpr int steps = 4;
    BeamArray beam(beamWidth, &workspace);
    beam.allocateHistory(steps + 1, beamWidth);

    // Simulate a simple beam search with known prevIndices and tokenIds
    // For each step, fill devicePrevIndices_ and deviceTokenIds_ with known values
    std::vector<int> prevs = { -1, 0, 1, 2, 1, 0, 2, 1, 0, 2, 1, 0 };
    std::vector<int> tokens = { 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42 };
    // Ensure prevs and tokens are sized for (steps+1)*beamWidth
    prevs.resize((steps + 1) * beamWidth);
    tokens.resize((steps + 1) * beamWidth);
    // Set last step's values to valid, meaningful values
    prevs[12] = 0; prevs[13] = 1; prevs[14] = 2;
    tokens[12] = 100; tokens[13] = 110; tokens[14] = 161;
    for (int s = 0; s < steps; ++s) {
        cudaMemcpy(beam.prevIndexPtr(), prevs.data() + s * beamWidth, beamWidth * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(beam.tokenIdPtr(), tokens.data() + s * beamWidth, beamWidth * sizeof(int), cudaMemcpyHostToDevice);
        beam.recordHistoryStep(beamWidth);
    }
    // Last step
    cudaMemcpy(beam.prevIndexPtr(), prevs.data() + steps * beamWidth, beamWidth * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(beam.tokenIdPtr(), tokens.data() + steps * beamWidth, beamWidth * sizeof(int), cudaMemcpyHostToDevice);
    beam.recordHistoryStep(beamWidth);

    // Reconstruct
    std::vector<int> out(beamWidth * (steps + 1));
    beam.reconstructHistory(out.data(), beamWidth, steps);

    // CPU reference: reconstruct for each beam
    for (int b = 0; b < beamWidth; ++b) {
        int idx = b;
        for (int s = steps; s >= 0; --s) {
            int token = tokens[s * beamWidth + idx];
            assert(out[b * (steps + 1) + s] == token);
            idx = prevs[s * beamWidth + idx];
            if (idx < 0) break;
        }
    }
    std::cout << "BeamArray GPU history reconstruction test passed!" << std::endl;
}

int main() {
    try {
        test_beam_array_basic();
        test_beam_array_batch();
        test_beam_array_sort_prune();
        test_beam_array_copy_to_host();
        test_beam_array_capacity();
        test_beam_array_history_reconstruct();
        std::cout << "All BeamArray tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 