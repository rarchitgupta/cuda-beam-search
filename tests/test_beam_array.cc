#include "whisper/beam_search/beam_array.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace whisper::beam_search;

void test_beam_array_basic() {
    BeamSearchWorkspace workspace(1024 * 1024);
    BeamArray beam(128, &workspace);
    
    assert(beam.Size() == 0);
    assert(beam.Capacity() >= 128);
    
    Token t1(1.5f, 42, 10);
    int idx = beam.AddToken(t1);
    assert(idx == 0);
    assert(beam.Size() == 1);
    
    Token t2 = beam.GetToken(0);
    assert(t2.score == 1.5f);
    assert(t2.token_id == 42);
    assert(t2.prev_index == 10);
    
    std::cout << "Basic BeamArray tests passed!" << std::endl;
}

void test_beam_array_batch() {
    BeamSearchWorkspace workspace(1024 * 1024);
    BeamArray beam(128, &workspace);
    
    std::vector<Token> tokens;
    for (int i = 0; i < 64; i++) {
        tokens.push_back(Token(static_cast<float>(i), i * 2, i - 1));
    }
    
    std::cout << "Attempting to add " << tokens.size() << " tokens to beam" << std::endl;
    int count = beam.AddTokens(tokens.data(), tokens.size());
    std::cout << "AddTokens returned count: " << count << ", beam.Size(): " << beam.Size() << std::endl;
    
    assert(count == 64);
    assert(beam.Size() == 64);
    
    for (int i = 0; i < 64; i++) {
        Token t = beam.GetToken(i);
        assert(t.score == static_cast<float>(i));
        assert(t.token_id == i * 2);
        assert(t.prev_index == i - 1);
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
    
    beam.AddTokens(tokens.data(), tokens.size());
    assert(beam.Size() == 100);
    
    beam.SortByScore();
    
    Token prev = beam.GetToken(0);
    for (int i = 1; i < 100; i++) {
        Token current = beam.GetToken(i);
        assert(prev.score >= current.score);
        prev = current;
    }
    
    beam.Prune(10);
    assert(beam.Size() == 10);
    
    std::sort(scores.begin(), scores.end(), std::greater<float>());
    for (int i = 0; i < 10; i++) {
        Token t = beam.GetToken(i);
        assert(t.score == scores[i]);
    }
    
    std::cout << "Sort and prune tests passed!" << std::endl;
}

void test_beam_array_copy_to_host() {
    BeamSearchWorkspace workspace(1024 * 1024);
    BeamArray beam(128, &workspace);
    
    for (int i = 0; i < 50; i++) {
        beam.AddToken(Token(static_cast<float>(i), i, i - 1));
    }
    
    std::vector<Token> host_tokens;
    beam.CopyToHost(host_tokens);
    
    assert(host_tokens.size() == 50);
    for (int i = 0; i < 50; i++) {
        assert(host_tokens[i].score == static_cast<float>(i));
        assert(host_tokens[i].token_id == i);
        assert(host_tokens[i].prev_index == i - 1);
    }
    
    std::cout << "Copy to host tests passed!" << std::endl;
}

void test_beam_array_capacity() {
    BeamSearchWorkspace workspace(1024 * 1024);
    BeamArray beam(10, &workspace);
    
    assert(beam.Capacity() >= 10);
    
    std::vector<Token> tokens(20);
    for (int i = 0; i < 20; i++) {
        tokens[i] = Token(static_cast<float>(i), i, i - 1);
    }
    
    int count = beam.AddTokens(tokens.data(), tokens.size());
    assert(count == 20);
    assert(beam.Size() == 20);
    assert(beam.Capacity() >= 20);
    
    std::cout << "Capacity growth tests passed!" << std::endl;
}

int main() {
    try {
        test_beam_array_basic();
        test_beam_array_batch();
        test_beam_array_sort_prune();
        test_beam_array_copy_to_host();
        test_beam_array_capacity();
        
        std::cout << "All BeamArray tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 