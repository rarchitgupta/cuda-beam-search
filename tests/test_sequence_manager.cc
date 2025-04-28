#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_array.h"
#include "whisper/beam_search/sequence_manager.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace whisper::beam_search;

void TestBasicSequenceReconstruction() {
    BeamSearchWorkspace workspace(10 * 1024 * 1024);
    
    // Create a sequence manager
    int max_batch_size = 2;
    int max_seq_length = 10;
    SequenceManager manager(&workspace, max_batch_size, max_seq_length);
    
    // Create beam arrays for positions
    BeamArray beam_pos0(5, &workspace);
    BeamArray beam_pos1(5, &workspace);
    BeamArray beam_pos2(5, &workspace);
    
    // Position 0 (first tokens)
    beam_pos0.AddToken(Token(-0.5f, 10, -1));  // Start token
    beam_pos0.AddToken(Token(-0.7f, 11, -1));  // Start token
    
    // Position 1 (second tokens)
    beam_pos1.AddToken(Token(-1.2f, 20, 0));   // From token 10
    beam_pos1.AddToken(Token(-1.5f, 21, 0));   // From token 10
    beam_pos1.AddToken(Token(-1.8f, 22, 1));   // From token 11
    
    // Position 2 (third tokens)
    beam_pos2.AddToken(Token(-2.2f, 30, 0));   // From token 20
    beam_pos2.AddToken(Token(-2.5f, 31, 1));   // From token 21
    
    // Track tokens
    int batch_index = 0;
    manager.TrackTokens(batch_index, 0, &beam_pos0);
    manager.TrackTokens(batch_index, 1, &beam_pos1);
    manager.TrackTokens(batch_index, 2, &beam_pos2);
    
    // Get best sequence
    std::vector<int> best_sequence = manager.GetBestSequence(batch_index);
    
    // Expected sequence: 10 (start) -> 20 -> 30
    assert(best_sequence.size() == 3);
    assert(best_sequence[0] == 10);
    assert(best_sequence[1] == 20);
    assert(best_sequence[2] == 30);
    
    // Get n-best sequences
    std::vector<std::vector<int>> nbest = manager.GetNBestSequences(batch_index, 2);
    
    assert(nbest.size() == 2);
    
    // First sequence should match best_sequence
    assert(nbest[0] == best_sequence);
    
    // Second sequence should be: 10 (start) -> 21 -> 31
    assert(nbest[1].size() == 3);
    assert(nbest[1][0] == 10);
    assert(nbest[1][1] == 21);
    assert(nbest[1][2] == 31);
    
    std::cout << "TestBasicSequenceReconstruction passed!" << std::endl;
}

void TestMultipleBatches() {
    BeamSearchWorkspace workspace(10 * 1024 * 1024);
    
    // Create a sequence manager
    int max_batch_size = 2;
    int max_seq_length = 10;
    SequenceManager manager(&workspace, max_batch_size, max_seq_length);
    
    // Create beam arrays for positions
    BeamArray beam1_pos0(5, &workspace);
    BeamArray beam1_pos1(5, &workspace);
    BeamArray beam2_pos0(5, &workspace);
    BeamArray beam2_pos1(5, &workspace);
    
    // Batch 0
    beam1_pos0.AddToken(Token(-0.5f, 10, -1));  // Start token
    beam1_pos1.AddToken(Token(-1.0f, 20, 0));   // From token 10
    
    // Batch 1
    beam2_pos0.AddToken(Token(-0.3f, 50, -1));  // Start token
    beam2_pos1.AddToken(Token(-0.8f, 60, 0));   // From token 50
    
    // Track tokens for both batches
    manager.TrackTokens(0, 0, &beam1_pos0);
    manager.TrackTokens(0, 1, &beam1_pos1);
    manager.TrackTokens(1, 0, &beam2_pos0);
    manager.TrackTokens(1, 1, &beam2_pos1);
    
    // Get sequences for both batches
    std::vector<int> seq1 = manager.GetBestSequence(0);
    std::vector<int> seq2 = manager.GetBestSequence(1);
    
    // Verify sequences
    assert(seq1.size() == 2);
    assert(seq1[0] == 10);
    assert(seq1[1] == 20);
    
    assert(seq2.size() == 2);
    assert(seq2[0] == 50);
    assert(seq2[1] == 60);
    
    // Test clearing a single batch
    manager.ClearBatch(0);
    
    // Batch 0 should be empty, but batch 1 should still have data
    assert(manager.GetBestSequence(0).empty());
    assert(!manager.GetBestSequence(1).empty());
    
    // Test resetting all batches
    manager.Reset();
    
    // Both batches should be empty now
    assert(manager.GetBestSequence(0).empty());
    assert(manager.GetBestSequence(1).empty());
    
    std::cout << "TestMultipleBatches passed!" << std::endl;
}

void TestCapacityGrowth() {
    BeamSearchWorkspace workspace(10 * 1024 * 1024);
    
    // Create a sequence manager
    int max_batch_size = 1;
    int max_seq_length = 10;
    SequenceManager manager(&workspace, max_batch_size, max_seq_length);
    
    // Start with a small beam
    BeamArray small_beam(2, &workspace);
    small_beam.AddToken(Token(-0.5f, 10, -1));
    small_beam.AddToken(Token(-0.7f, 11, -1));
    
    // Track with small beam
    manager.TrackTokens(0, 0, &small_beam);
    
    // Now use a larger beam
    BeamArray large_beam(5, &workspace);
    large_beam.AddToken(Token(-1.0f, 20, 0));
    large_beam.AddToken(Token(-1.2f, 21, 0));
    large_beam.AddToken(Token(-1.4f, 22, 0));
    large_beam.AddToken(Token(-1.6f, 23, 1));
    large_beam.AddToken(Token(-1.8f, 24, 1));
    
    // Track with larger beam
    manager.TrackTokens(0, 1, &large_beam);
    
    // Verify we can get sequences
    std::vector<std::vector<int>> sequences = manager.GetNBestSequences(0, 5);
    
    assert(sequences.size() == 5);
    
    // First sequence should be 10 -> 20
    assert(sequences[0].size() == 2);
    assert(sequences[0][0] == 10);
    assert(sequences[0][1] == 20);
    
    std::cout << "TestCapacityGrowth passed!" << std::endl;
}

int main() {
    try {
        TestBasicSequenceReconstruction();
        TestMultipleBatches();
        TestCapacityGrowth();
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
} 