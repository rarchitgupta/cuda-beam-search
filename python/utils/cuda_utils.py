#!/usr/bin/env python3
"""
Utility functions for working with CUDA tensors and beam search.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

from config import LOGITS_DIR, BUILD_DIR, MODULE_NAME, get_build_instructions

# Try to import our CUDA beam search module
try:
    import cuda_beam_search
except ImportError:
    # Add build directory to Python path
    sys.path.append(str(BUILD_DIR))
    try:
        import cuda_beam_search
    except ImportError:
        print(f"Failed to import {MODULE_NAME} module.")
        print(get_build_instructions())
        cuda_beam_search = None

def check_cuda_availability():
    """
    Check if CUDA is available and print device information
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    cuda_available = torch.cuda.is_available()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    return cuda_available

def load_logits_from_file(filename):
    """
    Load logits from a binary file saved by extract_logits.py
    
    Args:
        filename: Path to the binary file
    
    Returns:
        PyTorch tensor with the logits
    """
    # Resolve path - try standard location if not found
    if not os.path.exists(filename):
        alt_path = LOGITS_DIR / os.path.basename(filename)
        if os.path.exists(alt_path):
            filename = alt_path
        else:
            raise FileNotFoundError(f"Logits file not found: {filename}")
    
    # Try to load metadata
    meta_file = f"{filename}.meta"
    shape = None
    
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            for line in f:
                # Look for shape information
                if line.lower().startswith("shape:"):
                    shape_str = line.split(":", 1)[1].strip()
                    shape_str = shape_str.replace("[", "").replace("]", "")
                    shape = [int(x.strip()) for x in shape_str.split(",")]
                    if len(shape) == 2:  # Add batch dimension if missing
                        shape = [1] + shape
    
    # Load the binary file
    logits_np = np.fromfile(filename, dtype=np.float32)
    
    # Reshape using metadata or infer shape
    if shape is not None:
        total_size = np.prod(shape)
        if logits_np.size == total_size:
            logits_np = logits_np.reshape(shape)
        else:
            print(f"Warning: File size doesn't match metadata shape. Expected {total_size}, got {logits_np.size}")
            shape = None
    
    # If we don't have valid shape info, infer from file size
    if shape is None:
        # Most common case: single sequence step with large vocabulary
        shape = [1, 1, logits_np.size]
        print(f"No valid shape metadata found. Using shape: {shape}")
    
    print(f"Using shape: {shape}")
    logits_np = logits_np.reshape(shape)
    
    # Convert to PyTorch tensor and move to GPU if available
    logits_tensor = torch.from_numpy(logits_np).float()
    if torch.cuda.is_available():
        logits_tensor = logits_tensor.cuda()
    
    return logits_tensor

def create_random_logits(batch_size=1, seq_len=1, vocab_size=50257, device=None):
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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random logits
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    return logits

def test_tensor_bridge(logits_tensor):
    """
    Test the TensorBridge with the provided logits tensor
    
    Args:
        logits_tensor: PyTorch tensor containing logits
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if module is available
    if cuda_beam_search is None:
        print("Cannot test tensor bridge - CUDA beam search module not loaded")
        return False
    
    # Ensure tensor is on CUDA
    if not logits_tensor.is_cuda:
        if torch.cuda.is_available():
            print("Converting tensor to CUDA")
            logits_tensor = logits_tensor.cuda()
        else:
            print("CUDA not available - tensor bridge requires CUDA")
            return False
    
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