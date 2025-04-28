#pragma once

#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_array.h"
#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace whisper {
namespace beam_search {

/**
 * Manages the history of beam search sequences.
 * 
 * Allows for tracking token histories across multiple timesteps,
 * with support for multiple batches and beam widths.
 */
class SequenceManager {
public:
    SequenceManager(BeamSearchWorkspace* workspace, int max_batch_size = 1, int max_seq_length = 256);
    ~SequenceManager();
    
    // Allocate memory for a batch (call before tracking tokens)
    void AllocateBatchMemory(int batch_index);
    
    // Ensure capacity is sufficient for tracking tokens
    void EnsureCapacity(int batch_index, int position, int beam_width);
    
    // Track tokens at a specific position (blocking copy)
    void TrackTokens(int batch_index, int position, const BeamArray* beam);
    
    // Track tokens at a specific position (non-blocking copy)
    void TrackTokensAsync(int batch_index, int position, const BeamArray* beam, cudaStream_t stream);
    
    // Reconstruct a sequence by following beam indices
    std::vector<int> ReconstructSequence(int batch_index, int final_pos, int beam_index);
    
    // Get the best (highest scoring) sequence for a batch
    std::vector<int> GetBestSequence(int batch_index);
    
    // Get the top-n best sequences for a batch
    std::vector<std::vector<int>> GetNBestSequences(int batch_index, int n);
    
    // Clear a batch's history
    void ClearBatch(int batch_index);
    
    // Reset all batch histories
    void Reset();
    
private:
    BeamSearchWorkspace* workspace_;
    int max_batch_size_;
    int max_seq_length_;
    
    // Device memory for each batch
    int** d_token_history_;    // [batch][position * beam_width + beam_idx]
    int** d_prev_indices_;     // [batch][position * beam_width + beam_idx]
    float** d_scores_;         // [batch][position * beam_width + beam_idx]
    
    // Host shadow copies
    std::vector<std::vector<std::vector<int>>> h_token_history_;    // [batch][position][beam_idx]
    std::vector<std::vector<std::vector<int>>> h_prev_indices_;     // [batch][position][beam_idx]
    std::vector<std::vector<std::vector<float>>> h_scores_;         // [batch][position][beam_idx]
    
    // Track beam width history for each batch and position
    std::vector<std::vector<int>> beam_width_history_;  // [batch][position]
    
    // Track current sequence length for each batch
    std::vector<int> current_lengths_;  // [batch]
    
    // Helper methods
    void validateIndices(int batch_index, int position) const;
    
    void copyBeamToDeviceHistory(int batch_index, size_t offset, size_t size, 
                               const int* token_ids, const int* prev_indices, const float* scores);
    
    void copyBeamToDeviceHistoryAsync(int batch_index, size_t offset, size_t size, 
                                    const int* token_ids, const int* prev_indices, const float* scores,
                                    cudaStream_t stream);
    
    void copyBeamToHost(int batch_index, int position, const BeamArray* beam);
};

} // namespace beam_search
} // namespace whisper 