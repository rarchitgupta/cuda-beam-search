#include "whisper/beam_search/beam_types.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace whisper::beam_search;

void test_token() {
    // Create tokens
    Token t1;
    Token t2(1.5f, 42, 10);
    
    // Check default values
    assert(t1.score == 0.0f);
    assert(t1.tokenId == 0);
    assert(t1.prevIndex == -1);
    
    // Check custom values
    assert(t2.score == 1.5f);
    assert(t2.tokenId == 42);
    assert(t2.prevIndex == 10);
    
    std::cout << "Token tests passed!" << std::endl;
}

void test_workspace() {
    // Create workspace
    BeamSearchWorkspace workspace(1024); // Small initial size for testing
    
    // Allocate memory
    int* d_first = static_cast<int*>(workspace.allocate(256));
    float* d_second = static_cast<float*>(workspace.allocate(512));
    
    // Check memory usage
    assert(workspace.usedSize() >= 768);
    assert(workspace.usedSize() <= 1024);
    
    // Test workspace growth
    char* d_large = static_cast<char*>(workspace.allocate(2048));
    assert(workspace.capacity() >= 3072);
    
    // Reset and check
    workspace.reset();
    assert(workspace.usedSize() == 0);
    
    std::cout << "Workspace tests passed!" << std::endl;
}

int main() {
    try {
        test_token();
        test_workspace();
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 