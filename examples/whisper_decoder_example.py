#!/usr/bin/env python3
"""
Example script demonstrating the usage of the WhisperBeamSearchDecoder class.
This script shows how to transcribe audio using the CUDA-accelerated beam search.
"""

import torch
import time
import argparse
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sys
import os

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from python.whisper.beam_search_decoder import WhisperBeamSearchDecoder


def transcribe_with_cuda_beam_search(audio_input, model_name="openai/whisper-tiny.en", device=None,
                                     beam_width=5, top_k=50, top_p=1.0, temperature=1.0):
    """
    Transcribe audio using the CUDA-accelerated beam search decoder.
    
    Args:
        audio_input: Audio input dictionary with 'array' and 'sampling_rate' keys
        model_name: Name of the Whisper model to use
        device: Device to run on (cuda or cpu)
        beam_width: Beam width for beam search
        top_k: Top-k for filtering
        top_p: Top-p for nucleus sampling
        temperature: Temperature for generation
        
    Returns:
        dict: Dictionary with transcription and timing information
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and processor
    print(f"Loading Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Process audio
    print("Processing audio...")
    input_features = processor(
        audio_input["array"], 
        sampling_rate=audio_input["sampling_rate"], 
        return_tensors="pt"
    ).input_features.to(device)
    
    # Create decoder
    print("Creating beam search decoder...")
    decoder = WhisperBeamSearchDecoder(model, processor)
    
    # Configure beam search
    decoder.beam_config.beam_width = beam_width
    decoder.beam_config.top_k = top_k
    decoder.beam_config.top_p = top_p
    decoder.beam_config.temperature = temperature
    
    # Measure time for our beam search decoder
    print("Transcribing with CUDA beam search...")
    start_time = time.time()
    
    # Decode with CUDA beam search
    pred_ids = decoder.decode(
        input_features,
        language="en",
        task="transcribe"
    )
    
    beam_search_time = time.time() - start_time
    
    # Decode to text
    transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    # For comparison, run standard Whisper generation
    print("Transcribing with standard Whisper generation...")
    start_time = time.time()
    
    # Standard Whisper generation
    with torch.no_grad():
        standard_pred_ids = model.generate(
            input_features,
            num_beams=beam_width,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
    
    standard_time = time.time() - start_time
    
    # Decode standard output
    standard_transcription = processor.batch_decode(standard_pred_ids, skip_special_tokens=True)
    
    return {
        "cuda_beam_search": {
            "transcription": transcription[0],
            "time": beam_search_time
        },
        "standard_whisper": {
            "transcription": standard_transcription[0],
            "time": standard_time
        },
        "speedup": standard_time / beam_search_time if beam_search_time > 0 else 0
    }


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Whisper CUDA Beam Search Decoder Example")
    parser.add_argument("--model", type=str, default="openai/whisper-tiny.en", 
                        help="Whisper model to use")
    parser.add_argument("--beam_width", type=int, default=5, 
                        help="Beam width for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, 
                        help="Top-k for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, 
                        help="Top-p for nucleus sampling")
    args = parser.parse_args()
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: CUDA not available. This example requires a GPU for the best performance.")
    
    # Load sample audio
    print("Loading sample audio...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio_sample = ds[0]["audio"]
    
    # Transcribe
    result = transcribe_with_cuda_beam_search(
        audio_sample,
        model_name=args.model,
        device=device,
        beam_width=args.beam_width,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature
    )
    
    # Print results
    print("\n=== Results ===")
    print(f"CUDA Beam Search Transcription: {result['cuda_beam_search']['transcription']}")
    print(f"CUDA Beam Search Time: {result['cuda_beam_search']['time']:.4f} seconds")
    print(f"Standard Whisper Transcription: {result['standard_whisper']['transcription']}")
    print(f"Standard Whisper Time: {result['standard_whisper']['time']:.4f} seconds")
    print(f"Speedup: {result['speedup']:.2f}x")


if __name__ == "__main__":
    main() 