#!/usr/bin/env python3
# Copyright (c) 2023. All rights reserved.

"""
Example script demonstrating integration of CUDA beam search with Whisper model.
This example shows how to:
1. Load a Whisper model
2. Process audio input
3. Use our CUDA-accelerated beam search for decoding
4. Compare performance with Whisper's native generation
"""

import argparse
import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from cuda_beam_search import TensorBridge, BeamSearchConfig
import numpy as np


def transcribe_with_default_whisper(model, processor, audio_input, device):
    """Transcribe audio using Whisper's default generation method."""
    start_time = time.time()
    
    # Process audio input
    input_features = processor(
        audio_input["array"], 
        sampling_rate=audio_input["sampling_rate"], 
        return_tensors="pt"
    ).input_features.to(device)

    # Generate token ids with default method
    predicted_ids = model.generate(input_features)
    
    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    end_time = time.time()
    
    return {
        "transcription": transcription,
        "time_taken": end_time - start_time
    }


def transcribe_with_cuda_beam_search(model, processor, audio_input, device, beam_config):
    """Transcribe audio using our CUDA-accelerated beam search."""
    start_time = time.time()
    
    # Process audio input
    input_features = processor(
        audio_input["array"], 
        sampling_rate=audio_input["sampling_rate"], 
        return_tensors="pt"
    ).input_features.to(device)
    
    # Get encoder outputs
    encoder_outputs = model.get_encoder()(input_features)
    
    # Initialize with language and task tokens
    # This is Whisper-specific initialization
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device)
    
    # Create TensorBridge instance
    tensor_bridge = TensorBridge()
    
    # Start decoding loop
    max_length = beam_config.max_length
    decoded_ids = []
    
    for i in range(max_length):
        # Get decoder outputs for current position
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0]
        )
        
        # Get logits from decoder outputs
        logits = model.proj_out(decoder_outputs[0])
        
        # Focus on the last token position's logits
        next_token_logits = logits[:, -1, :]
        
        # Set logits in TensorBridge
        tensor_bridge.set_logits_tensor(next_token_logits)
        
        # Execute beam search
        tensor_bridge.execute_beam_search(beam_config)
        
        # Get beam search results
        token_sequences = tensor_bridge.get_beam_search_results()
        
        # For simplicity, we're just taking the best sequence
        next_token_id = token_sequences[0][-1] if token_sequences and token_sequences[0] else model.config.eos_token_id
        
        # Add to decoded sequence
        decoded_ids.append(next_token_id)
        
        # Update decoder input ids for next iteration
        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)
        
        # Stop if we generated the EOS token
        if next_token_id == model.config.eos_token_id:
            break
    
    # Convert to tensor format expected by processor
    predicted_ids = torch.tensor([decoded_ids], device=device)
    
    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    end_time = time.time()
    
    return {
        "transcription": transcription,
        "time_taken": end_time - start_time
    }


def benchmark_memory_usage(model, processor, audio_input, device, beam_config):
    """Benchmark memory usage for both decoding methods."""
    # Record baseline memory
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    baseline_memory = torch.cuda.max_memory_allocated(device)
    
    # Default Whisper generation
    torch.cuda.reset_peak_memory_stats(device)
    transcribe_with_default_whisper(model, processor, audio_input, device)
    whisper_memory = torch.cuda.max_memory_allocated(device) - baseline_memory
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # CUDA beam search
    torch.cuda.reset_peak_memory_stats(device)
    transcribe_with_cuda_beam_search(model, processor, audio_input, device, beam_config)
    beam_search_memory = torch.cuda.max_memory_allocated(device) - baseline_memory
    
    return {
        "whisper_memory_bytes": whisper_memory,
        "beam_search_memory_bytes": beam_search_memory,
        "memory_reduction_percent": (1 - beam_search_memory / whisper_memory) * 100 if whisper_memory > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Whisper CUDA Beam Search Demo")
    parser.add_argument("--model", type=str, default="openai/whisper-tiny.en", help="Whisper model to use")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarking")
    args = parser.parse_args()
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available. This demo requires a GPU to run.")
        return
    
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading Whisper model: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    
    # Enable FP16 if available
    if hasattr(torch.cuda, "amp") and torch.cuda.is_available():
        model = model.half()
        print("Using FP16 precision")
    
    # Load sample audio
    print("Loading sample audio")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio_sample = ds[0]["audio"]
    
    # Configure beam search
    beam_config = BeamSearchConfig()
    beam_config.beam_width = args.beam_width
    beam_config.temperature = args.temperature
    beam_config.top_k = args.top_k
    beam_config.top_p = args.top_p
    beam_config.max_length = args.max_length
    
    # Run transcription with default Whisper generation
    print("\nRunning transcription with default Whisper generation...")
    whisper_result = transcribe_with_default_whisper(model, processor, audio_sample, device)
    print(f"Whisper transcription: {whisper_result['transcription'][0]}")
    print(f"Time taken: {whisper_result['time_taken']:.4f} seconds")
    
    # Run transcription with CUDA beam search
    print("\nRunning transcription with CUDA beam search...")
    beam_search_result = transcribe_with_cuda_beam_search(model, processor, audio_sample, device, beam_config)
    print(f"CUDA beam search transcription: {beam_search_result['transcription'][0]}")
    print(f"Time taken: {beam_search_result['time_taken']:.4f} seconds")
    
    # Show speedup
    speedup = whisper_result['time_taken'] / beam_search_result['time_taken'] if beam_search_result['time_taken'] > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Run benchmarking if requested
    if args.benchmark:
        print("\nRunning memory usage benchmark...")
        memory_results = benchmark_memory_usage(model, processor, audio_sample, device, beam_config)
        
        print(f"Whisper memory usage: {memory_results['whisper_memory_bytes'] / 1024 / 1024:.2f} MB")
        print(f"CUDA beam search memory usage: {memory_results['beam_search_memory_bytes'] / 1024 / 1024:.2f} MB")
        print(f"Memory reduction: {memory_results['memory_reduction_percent']:.2f}%")
        
        # Run multiple iterations for timing benchmark
        iterations = 10
        whisper_times = []
        beam_search_times = []
        
        print(f"\nRunning timing benchmark ({iterations} iterations)...")
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            whisper_result = transcribe_with_default_whisper(model, processor, audio_sample, device)
            beam_search_result = transcribe_with_cuda_beam_search(model, processor, audio_sample, device, beam_config)
            
            whisper_times.append(whisper_result['time_taken'])
            beam_search_times.append(beam_search_result['time_taken'])
        
        # Calculate statistics
        whisper_avg = np.mean(whisper_times)
        whisper_std = np.std(whisper_times)
        beam_search_avg = np.mean(beam_search_times)
        beam_search_std = np.std(beam_search_times)
        
        print("\nBenchmark results:")
        print(f"Whisper average time: {whisper_avg:.4f} ± {whisper_std:.4f} seconds")
        print(f"CUDA beam search average time: {beam_search_avg:.4f} ± {beam_search_std:.4f} seconds")
        print(f"Average speedup: {whisper_avg / beam_search_avg:.2f}x")


if __name__ == "__main__":
    main() 