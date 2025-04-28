#include "whisper/beam_search/sequence_manager.h"
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace whisper {
namespace beam_search {

SequenceManager::SequenceManager(BeamSearchWorkspace* workspace, int max_batch_size, int max_seq_length)
    : workspace_(workspace),
      max_batch_size_(max_batch_size), 
      max_seq_length_(max_seq_length) {
    
    // Allocate host-side data structures
    h_token_history_.resize(max_batch_size);
    h_prev_indices_.resize(max_batch_size);
    h_scores_.resize(max_batch_size);
    beam_width_history_.resize(max_batch_size);
    current_lengths_.resize(max_batch_size, 0);
    
    // Allocate device memory pointers for each batch
    d_token_history_ = new int*[max_batch_size];
    d_prev_indices_ = new int*[max_batch_size];
    d_scores_ = new float*[max_batch_size];
    
    // Initialize pointers to nullptr
    for (int i = 0; i < max_batch_size; i++) {
        d_token_history_[i] = nullptr;
        d_prev_indices_[i] = nullptr;
        d_scores_[i] = nullptr;
    }
}

SequenceManager::~SequenceManager() {
    // Free device memory
    for (int i = 0; i < max_batch_size_; i++) {
        if (d_token_history_[i]) {
            cudaFree(d_token_history_[i]);
        }
        if (d_prev_indices_[i]) {
            cudaFree(d_prev_indices_[i]);
        }
        if (d_scores_[i]) {
            cudaFree(d_scores_[i]);
        }
    }
    
    // Free pointer arrays
    delete[] d_token_history_;
    delete[] d_prev_indices_;
    delete[] d_scores_;
}

void SequenceManager::AllocateBatchMemory(int batch_index) {
    // Initial allocation for a reasonable beam width 
    // (we'll grow this if needed in EnsureCapacity)
    const int initial_beam_width = 5;
    EnsureCapacity(batch_index, 0, initial_beam_width);
}

void SequenceManager::EnsureCapacity(int batch_index, int position, int beam_width) {
    if (batch_index < 0 || batch_index >= max_batch_size_) {
        throw std::out_of_range("Batch index out of range");
    }
    
    if (position < 0 || position >= max_seq_length_) {
        throw std::out_of_range("Position out of range");
    }
    
    // Update sequence length if needed
    if (position >= current_lengths_[batch_index]) {
        current_lengths_[batch_index] = position + 1;
    }
    
    // Ensure host vectors are large enough
    if (h_token_history_[batch_index].size() <= position) {
        h_token_history_[batch_index].resize(position + 1);
        h_prev_indices_[batch_index].resize(position + 1);
        h_scores_[batch_index].resize(position + 1);
        beam_width_history_[batch_index].resize(position + 1, 0);
    }
    
    // Check if we already have enough capacity at this position
    if (beam_width_history_[batch_index][position] >= beam_width) {
        return;
    }
    
    // First-time allocation
    if (d_token_history_[batch_index] == nullptr) {
        size_t total_elements = max_seq_length_ * beam_width;
        
        // Allocate device memory for this batch
        cudaMalloc(&d_token_history_[batch_index], total_elements * sizeof(int));
        cudaMalloc(&d_prev_indices_[batch_index], total_elements * sizeof(int));
        cudaMalloc(&d_scores_[batch_index], total_elements * sizeof(float));
        
        // Initialize host vectors
        for (int i = 0; i <= position; i++) {
            h_token_history_[batch_index][i].resize(beam_width, 0);
            h_prev_indices_[batch_index][i].resize(beam_width, -1);
            h_scores_[batch_index][i].resize(beam_width, 0.0f);
        }
    } 
    // Need to expand existing allocation
    else if (beam_width > beam_width_history_[batch_index][position]) {
        // Calculate total elements needed
        size_t old_beam_width = beam_width_history_[batch_index][position];
        size_t new_beam_width = beam_width;
        size_t total_elements = max_seq_length_ * new_beam_width;
        
        // Allocate new memory
        int* new_token_history;
        int* new_prev_indices;
        float* new_scores;
        
        cudaMalloc(&new_token_history, total_elements * sizeof(int));
        cudaMalloc(&new_prev_indices, total_elements * sizeof(int));
        cudaMalloc(&new_scores, total_elements * sizeof(float));
        
        // Copy existing data to new memory
        for (int pos = 0; pos < current_lengths_[batch_index]; pos++) {
            size_t elements_to_copy = beam_width_history_[batch_index][pos];
            
            if (elements_to_copy > 0) {
                size_t old_offset = pos * old_beam_width;
                size_t new_offset = pos * new_beam_width;
                
                cudaMemcpy(new_token_history + new_offset, 
                          d_token_history_[batch_index] + old_offset,
                          elements_to_copy * sizeof(int), 
                          cudaMemcpyDeviceToDevice);
                
                cudaMemcpy(new_prev_indices + new_offset, 
                          d_prev_indices_[batch_index] + old_offset,
                          elements_to_copy * sizeof(int), 
                          cudaMemcpyDeviceToDevice);
                
                cudaMemcpy(new_scores + new_offset, 
                          d_scores_[batch_index] + old_offset,
                          elements_to_copy * sizeof(float), 
                          cudaMemcpyDeviceToDevice);
            }
        }
        
        // Free old memory
        cudaFree(d_token_history_[batch_index]);
        cudaFree(d_prev_indices_[batch_index]);
        cudaFree(d_scores_[batch_index]);
        
        // Update pointers
        d_token_history_[batch_index] = new_token_history;
        d_prev_indices_[batch_index] = new_prev_indices;
        d_scores_[batch_index] = new_scores;
        
        // Resize host vectors
        for (int i = 0; i <= position; i++) {
            h_token_history_[batch_index][i].resize(new_beam_width, 0);
            h_prev_indices_[batch_index][i].resize(new_beam_width, -1);
            h_scores_[batch_index][i].resize(new_beam_width, 0.0f);
        }
    }
    
    // Update beam width history
    beam_width_history_[batch_index][position] = beam_width;
}

void SequenceManager::TrackTokens(int batch_index, int position, const BeamArray* beam) {
    validateIndices(batch_index, position);
    
    // Get beam size
    size_t beam_size = beam->Size();
    if (beam_size == 0) {
        return; // Nothing to track
    }
    
    // Ensure we have enough capacity
    EnsureCapacity(batch_index, position, beam_size);
    
    // Get device pointers
    float* d_beam_scores = beam->GetScorePtr();
    int* d_beam_token_ids = beam->GetTokenIdPtr();
    int* d_beam_prev_indices = beam->GetPrevIndexPtr();
    
    // Calculate offsets
    size_t history_offset = position * beam_width_history_[batch_index][position];
    
    // Copy beam data to history
    copyBeamToDeviceHistory(batch_index, history_offset, beam_size, 
                           d_beam_token_ids, d_beam_prev_indices, d_beam_scores);
    
    // Update host shadow copies
    copyBeamToHost(batch_index, position, beam);
}

void SequenceManager::TrackTokensAsync(int batch_index, int position, const BeamArray* beam, cudaStream_t stream) {
    validateIndices(batch_index, position);
    
    // Get beam size
    size_t beam_size = beam->Size();
    if (beam_size == 0) {
        return; // Nothing to track
    }
    
    // Ensure we have enough capacity
    EnsureCapacity(batch_index, position, beam_size);
    
    // Get device pointers
    float* d_beam_scores = beam->GetScorePtr();
    int* d_beam_token_ids = beam->GetTokenIdPtr();
    int* d_beam_prev_indices = beam->GetPrevIndexPtr();
    
    // Calculate offsets
    size_t history_offset = position * beam_width_history_[batch_index][position];
    
    // Copy beam data to history asynchronously using the provided stream
    copyBeamToDeviceHistoryAsync(batch_index, history_offset, beam_size, 
                               d_beam_token_ids, d_beam_prev_indices, d_beam_scores, stream);
    
    // Host shadow copies will be updated when needed
}

std::vector<int> SequenceManager::ReconstructSequence(int batch_index, int final_pos, int beam_index) {
    validateIndices(batch_index, final_pos);
    
    if (beam_index < 0 || beam_index >= beam_width_history_[batch_index][final_pos]) {
        throw std::out_of_range("Beam index out of range");
    }
    
    std::vector<int> sequence;
    int current_pos = final_pos;
    int current_beam = beam_index;
    
    // Backtrack through the sequence
    while (current_pos >= 0) {
        // Add token to sequence
        int token_id = h_token_history_[batch_index][current_pos][current_beam];
        sequence.push_back(token_id);
        
        // Get previous index
        int prev_index = h_prev_indices_[batch_index][current_pos][current_beam];
        
        // Stop if this is a start token (prev_index == -1)
        if (prev_index == -1) {
            break;
        }
        
        // Move to previous position
        current_beam = prev_index;
        current_pos--;
    }
    
    // Reverse the sequence (since we built it backwards)
    std::reverse(sequence.begin(), sequence.end());
    
    return sequence;
}

std::vector<int> SequenceManager::GetBestSequence(int batch_index) {
    if (batch_index < 0 || batch_index >= max_batch_size_) {
        throw std::out_of_range("Batch index out of range");
    }
    
    if (current_lengths_[batch_index] == 0) {
        return {}; // No sequence tracked yet
    }
    
    // Get final position
    int final_pos = current_lengths_[batch_index] - 1;
    
    // Beam with index 0 is the best one (highest score)
    return ReconstructSequence(batch_index, final_pos, 0);
}

std::vector<std::vector<int>> SequenceManager::GetNBestSequences(int batch_index, int n) {
    if (batch_index < 0 || batch_index >= max_batch_size_) {
        throw std::out_of_range("Batch index out of range");
    }
    
    if (current_lengths_[batch_index] == 0) {
        return {}; // No sequence tracked yet
    }
    
    // Get final position
    int final_pos = current_lengths_[batch_index] - 1;
    
    // Limit n to available beam width
    n = std::min(n, beam_width_history_[batch_index][final_pos]);
    
    std::vector<std::vector<int>> sequences;
    sequences.reserve(n);
    
    // Get n-best sequences
    for (int i = 0; i < n; i++) {
        sequences.push_back(ReconstructSequence(batch_index, final_pos, i));
    }
    
    return sequences;
}

void SequenceManager::ClearBatch(int batch_index) {
    if (batch_index < 0 || batch_index >= max_batch_size_) {
        throw std::out_of_range("Batch index out of range");
    }
    
    // Reset host data
    h_token_history_[batch_index].clear();
    h_prev_indices_[batch_index].clear();
    h_scores_[batch_index].clear();
    beam_width_history_[batch_index].clear();
    current_lengths_[batch_index] = 0;
}

void SequenceManager::Reset() {
    // Reset all batches
    for (int i = 0; i < max_batch_size_; i++) {
        ClearBatch(i);
    }
}

// Private helper methods

void SequenceManager::validateIndices(int batch_index, int position) const {
    if (batch_index < 0 || batch_index >= max_batch_size_) {
        throw std::out_of_range("Batch index out of range");
    }
    
    if (position < 0 || position >= max_seq_length_) {
        throw std::out_of_range("Position out of range");
    }
}

void SequenceManager::copyBeamToDeviceHistory(int batch_index, size_t offset, size_t size, 
                                             const int* token_ids, const int* prev_indices, const float* scores) {
    cudaMemcpy(d_token_history_[batch_index] + offset,
              token_ids,
              size * sizeof(int),
              cudaMemcpyDeviceToDevice);
    
    cudaMemcpy(d_prev_indices_[batch_index] + offset,
              prev_indices,
              size * sizeof(int),
              cudaMemcpyDeviceToDevice);
    
    cudaMemcpy(d_scores_[batch_index] + offset,
              scores,
              size * sizeof(float),
              cudaMemcpyDeviceToDevice);
}

void SequenceManager::copyBeamToDeviceHistoryAsync(int batch_index, size_t offset, size_t size, 
                                                 const int* token_ids, const int* prev_indices, const float* scores,
                                                 cudaStream_t stream) {
    cudaMemcpyAsync(d_token_history_[batch_index] + offset,
                   token_ids,
                   size * sizeof(int),
                   cudaMemcpyDeviceToDevice,
                   stream);
    
    cudaMemcpyAsync(d_prev_indices_[batch_index] + offset,
                   prev_indices,
                   size * sizeof(int),
                   cudaMemcpyDeviceToDevice,
                   stream);
    
    cudaMemcpyAsync(d_scores_[batch_index] + offset,
                   scores,
                   size * sizeof(float),
                   cudaMemcpyDeviceToDevice,
                   stream);
}

void SequenceManager::copyBeamToHost(int batch_index, int position, const BeamArray* beam) {
    std::vector<Token> host_tokens;
    beam->CopyToHost(host_tokens);
    
    for (size_t i = 0; i < host_tokens.size(); i++) {
        h_token_history_[batch_index][position][i] = host_tokens[i].token_id;
        h_prev_indices_[batch_index][position][i] = host_tokens[i].prev_index;
        h_scores_[batch_index][position][i] = host_tokens[i].score;
    }
}

} // namespace beam_search
} // namespace whisper 