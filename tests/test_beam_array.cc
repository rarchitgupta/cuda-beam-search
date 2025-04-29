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