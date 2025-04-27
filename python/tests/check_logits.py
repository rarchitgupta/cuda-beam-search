#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import argparse

def analyze_logits_file(file_path):
    """
    Analyze a logits binary file to determine its shape and content
    
    Args:
        file_path: Path to the binary file containing logits
    """
    print(f"Analyzing logits file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return
    
    # Load the binary file as numpy array
    try:
        logits_np = np.fromfile(file_path, dtype=np.float32)
        print(f"Successfully loaded file with {logits_np.size} elements (dtype: {logits_np.dtype})")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Check if it's a valid size for Whisper model
    whisper_vocab_size = 50257
    
    # Print statistics about the array
    print(f"\nData statistics:")
    print(f"  Min value: {logits_np.min()}")
    print(f"  Max value: {logits_np.max()}")
    print(f"  Mean value: {logits_np.mean()}")
    print(f"  Standard deviation: {logits_np.std()}")
    
    # Try to determine the shape
    print(f"\nPossible shapes:")
    
    # Option 1: Default shape (single token prediction)
    print(f"  1. Default shape (single token prediction): (1, 1, {logits_np.size})")
    
    # Option 2: Whisper vocabulary size
    if logits_np.size % whisper_vocab_size == 0:
        seq_len = logits_np.size // whisper_vocab_size
        print(f"  2. Whisper compatible shape: (1, {seq_len}, {whisper_vocab_size})")
        
        # Reshape and analyze
        try:
            reshaped = logits_np.reshape(1, seq_len, whisper_vocab_size)
            print(f"     Successfully reshaped to (1, {seq_len}, {whisper_vocab_size})")
            
            # Get top tokens
            indices = np.argsort(-reshaped[0, 0, :])[:10]
            values = reshaped[0, 0, indices]
            print("\nTop 10 logits in first position:")
            for i, (idx, val) in enumerate(zip(indices, values)):
                print(f"  {i+1}. Index {idx}: {val}")
                
            # Check if it looks like a proper distribution
            softmax = np.exp(reshaped[0, 0, :]) / np.sum(np.exp(reshaped[0, 0, :]))
            entropy = -np.sum(softmax * np.log(softmax + 1e-10))
            print(f"\nEntropy of first position distribution: {entropy}")
            print(f"Max probability: {softmax.max()}")
            if entropy > 1.0 and softmax.max() < 0.9:
                print("✓ Distribution looks reasonable (not too concentrated)")
            else:
                print("⚠ Distribution might be too concentrated or uniform")
        except Exception as e:
            print(f"     Error reshaping: {e}")
    else:
        print(f"  ⚠ Size {logits_np.size} is not divisible by Whisper vocab size {whisper_vocab_size}")
        print(f"     Remainder: {logits_np.size % whisper_vocab_size}")
    
    # Check if usable with our tensor bridge
    print("\nCompatibility with tensor bridge:")
    
    # Convert to PyTorch tensor
    try:
        # Create tensor on CPU first
        logits_tensor = torch.from_numpy(logits_np.copy()).float()
        print(f"✓ Can convert to PyTorch tensor: {logits_tensor.shape}")
        
        # Check if we can move to CUDA
        if torch.cuda.is_available():
            try:
                logits_tensor = logits_tensor.cuda()
                print(f"✓ Can move tensor to CUDA device")
                
                # Check if tensor is contiguous
                if logits_tensor.is_contiguous():
                    print(f"✓ Tensor is contiguous")
                else:
                    print(f"⚠ Tensor is not contiguous - would need to call .contiguous()")
                
                # Try reshaping if it's compatible with Whisper
                if logits_np.size % whisper_vocab_size == 0:
                    seq_len = logits_np.size // whisper_vocab_size
                    try:
                        logits_tensor = logits_tensor.reshape(1, seq_len, whisper_vocab_size)
                        print(f"✓ Can reshape to (1, {seq_len}, {whisper_vocab_size})")
                        print(f"✓ Final tensor shape: {logits_tensor.shape}")
                        print("\nConclusion: This file is compatible with our tensor bridge.")
                    except Exception as e:
                        print(f"⚠ Error reshaping tensor: {e}")
            except Exception as e:
                print(f"⚠ Error moving tensor to CUDA: {e}")
        else:
            print(f"⚠ CUDA not available for testing GPU compatibility")
    except Exception as e:
        print(f"⚠ Error converting to PyTorch tensor: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze logits binary file")
    parser.add_argument("--file", default="../logits.bin", help="Path to logits binary file")
    
    args = parser.parse_args()
    analyze_logits_file(args.file)

if __name__ == "__main__":
    main() 