#pragma once

#include <cstddef>
#include <vector>

#include "whisper/beam_search/beam_types.h"

namespace whisper {
namespace beam_search {

// Manages beam tokens using Structure of Arrays (SoA) pattern for efficient GPU memory access.
class BeamArray {
public:
    // Constructs a BeamArray with the given maximum beam size and workspace.
    BeamArray(std::size_t maxBeamSize, BeamSearchWorkspace* workspace);

    ~BeamArray();

    // Resets the array to an empty state.
    void reset();

    // Returns the current number of tokens.
    std::size_t size() const;

    // Returns the capacity (maximum beam size).
    std::size_t capacity() const;

    // Adds a single token; returns its index or -1 on failure.
    int addToken(const Token& token);

    // Adds multiple tokens; returns the number actually added.
    int addTokens(const Token* tokens, std::size_t count);

    // Sorts tokens by descending score.
    void sortByScore();

    // Keeps only the top 'beamWidth' tokens by score.
    void prune(std::size_t beamWidth);

    // Retrieves the token at the given index.
    Token getToken(std::size_t index) const;

    // Copies tokens to the provided host vector.
    void copyToHost(std::vector<Token>& hostTokens) const;

    // Device pointers for direct access.
    float* scorePtr() const { return deviceScores_; }
    int* tokenIdPtr() const { return deviceTokenIds_; }
    int* prevIndexPtr() const { return devicePrevIndices_; }

    // --- GPU-based history reconstruction ---
    // Allocates device history buffers for up to maxSteps steps and beamWidth beams.
    void allocateHistory(std::size_t maxSteps, std::size_t beamWidth);
    // Records the current step's backpointers and tokenIds to device history.
    void recordHistoryStep(std::size_t beamWidth);
    // Reconstructs the best path for each beam on the GPU and copies to host.
    // hostOutput: shape [beamWidth][stepCount+1], row-major.
    void reconstructHistory(int* hostOutput, std::size_t beamWidth, std::size_t stepCount);

private:
    // Ensures there is capacity for the required number of tokens.
    void ensureCapacity(std::size_t requiredSize);

    // Allocates device memory.
    void allocateMemory();

    // Copies raw arrays to host vectors.
    void copyToHost(std::vector<float>& scores,
                    std::vector<int>& tokenIds,
                    std::vector<int>& prevIndices);

    // Device memory in SoA layout.
    float* deviceScores_ = nullptr;
    int* deviceTokenIds_ = nullptr;
    int* devicePrevIndices_ = nullptr;
    int* deviceIndices_ = nullptr;

    // Device-side history tracking
    int* d_historyPrevIndices_ = nullptr; // [maxSteps * beamWidth]
    int* d_historyTokenIds_ = nullptr;    // [maxSteps * beamWidth]
    std::size_t maxHistorySteps_ = 0;
    std::size_t historyStep_ = 0;

    // Host shadow copies for quick access.
    std::vector<float> hostScores_;
    std::vector<int> hostTokenIds_;
    std::vector<int> hostPrevIndices_;

    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
    BeamSearchWorkspace* workspace_ = nullptr;
};

} // namespace beam_search
} // namespace whisper