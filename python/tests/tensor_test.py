#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import argparse

# Import our utilities
from ..utils.cuda_utils import create_random_logits, load_logits_from_file, test_tensor_bridge

# Try to import our module - handle path issues
try:
    import cuda_beam_search
except ImportError:
    # Add the build directory to the path if the module is not installed
    from ..config import BUILD_DIR
    sys.path.append(str(BUILD_DIR))
    try:
        import cuda_beam_search
    except ImportError:
        print("Failed to import cuda_beam_search module. Make sure it's installed or built.")
        print("You can build it by running: cmake .. && make in the build directory")
        sys.exit(1)

def create_random_logits(batch_size=1, seq_len=10, vocab_size=50257, device='cuda'):
    """
    Create random logits tensor for testing
    
    Args:
        batch_size: Number of sequences in batch
        seq_len: Length of each sequence
        vocab_size: Size of vocabulary (number of possible tokens)
        device: Device to create tensor on ('cpu' or 'cuda')
    
    Returns:
        PyTorch tensor with random logits
    """
    # Check if CUDA is available when device is 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Create random logits
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    return logits

def test_tensor_bridge(logits_tensor):
    """
    Test the TensorBridge with the provided logits tensor
    
    Args:
        logits_tensor: PyTorch tensor containing logits
    """
    # Ensure tensor is on CUDA
    if not logits_tensor.is_cuda:
        print("Converting tensor to CUDA")
        logits_tensor = logits_tensor.cuda()
    
    # Create TensorBridge instance
    bridge = cuda_beam_search.TensorBridge()
    
    # Pass tensor to C++
    success = bridge.set_logits_tensor(logits_tensor)
    
    if success:
        print("Successfully passed tensor to C++")
        shape = bridge.get_shape()
        print(f"Shape reported by C++: {shape}")
        print(f"Shape in Python: {logits_tensor.shape}")
        
        # Verify shape matches
        assert shape[0] == logits_tensor.shape[0], "Batch size mismatch"
        assert shape[1] == logits_tensor.shape[1], "Sequence length mismatch" 
        assert shape[2] == logits_tensor.shape[2], "Vocabulary size mismatch"
        
        print("All shapes match! Tensor bridge is working correctly.")
        return True
    else:
        print("Failed to pass tensor to C++")
        return False

def load_logits_from_file(filename):
    """
    Load logits from a binary file saved by extract_logits.py
    
    Args:
        filename: Path to the binary file
    
    Returns:
        PyTorch tensor with the logits
    """
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Logits file not found: {filename}")
    
    # Try to load metadata
    meta_file = f"{filename}.meta"
    shape = None
    
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            for line in f:
                if line.startswith("Shape:"):
                    shape_str = line.replace("Shape:", "").strip()
                    shape_str = shape_str.replace("(", "").replace(")", "")
                    shape = [int(x.strip()) for x in shape_str.split(",")]
                    # If shape has only 2 dimensions, add batch dimension
                    if len(shape) == 2:
                        shape = [1] + shape
    
    # Load the binary file as numpy array
    logits_np = np.fromfile(filename, dtype=np.float32)
    
    # If we have metadata shape, use it
    if shape is not None:
        total_size = np.prod(shape)
        if logits_np.size == total_size:
            logits_np = logits_np.reshape(shape)
        else:
            print(f"Warning: File size doesn't match expected shape from metadata. Expected {total_size}, got {logits_np.size}")
            shape = None
    
    # If we don't have shape info or it was invalid, infer from file size
    if shape is None:
        # Assume batch_size = 1 for simplicity
        batch_size = 1
        
        # Try to determine sequence length and vocab size
        # First, check if it's a single sequence step (most common)
        if logits_np.size > 1000:  # Reasonable min vocab size
            seq_len = 1
            vocab_size = logits_np.size // batch_size // seq_len
            shape = [batch_size, seq_len, vocab_size]
        else:
            # For multi-sequence data, we'll need additional info
            print(f"Warning: Could not determine shape. File contains {logits_np.size} values.")
            
            # As a fallback, just reshape as [1, 1, size]
            shape = [1, 1, logits_np.size]
    
    print(f"Using shape: {shape}")
    logits_np = logits_np.reshape(shape)
    
    # Convert to PyTorch tensor and move to GPU
    logits_tensor = torch.from_numpy(logits_np).float()
    if torch.cuda.is_available():
        logits_tensor = logits_tensor.cuda()
    
    return logits_tensor

def main():
    parser = argparse.ArgumentParser(description="Test CUDA tensor bridge")
    parser.add_argument("--file", help="Path to logits binary file (from extract_logits.py)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (if generating random logits)")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length (if generating random logits)")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size (if generating random logits)")
    
    args = parser.parse_args()
    
    print("CUDA Beam Search Tensor Bridge Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Either load logits from file or generate random ones
    if args.file:
        print(f"Loading logits from file: {args.file}")
        logits = load_logits_from_file(args.file)
    else:
        print("Generating random logits")
        logits = create_random_logits(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size
        )
    
    print(f"Logits tensor shape: {logits.shape}")
    test_tensor_bridge(logits)

if __name__ == "__main__":
    main() 