#!/usr/bin/env python3
"""
Main entry point for CUDA beam search on Whisper logits.
Provides a complete pipeline for audio processing, logit extraction and CUDA testing.
"""
import os
import argparse
import time
from pathlib import Path

# Direct imports
from config import AUDIO_DIR, LOGITS_DIR, DEFAULT_MODEL_SIZE, DEFAULT_LANGUAGE
from whisper import load_audio, load_whisper_model, extract_logits, save_logits
from utils import cuda_utils

def process_audio(audio_path, output_path=None, model_name="large", language="en", 
                 device=None, skip_existing=False):
    """Process an audio file through the entire pipeline"""
    # Generate output path if not provided
    if output_path is None:
        audio_basename = Path(audio_path).stem
        output_path = LOGITS_DIR / f"{audio_basename}.logits.bin"
    
    # Skip if output already exists and requested
    if skip_existing and os.path.exists(output_path):
        print(f"Output file exists: {output_path}")
        print("Skipping extraction (--skip-existing is set)")
        return output_path
    
    # Full pipeline with timing for each step
    steps = [
        ("Loading Audio", lambda: load_audio(audio_path)),
        ("Loading Whisper Model", lambda: load_whisper_model(model_name, device)),
        ("Extracting Logits", lambda waveform, processor_model: 
            extract_logits(waveform, *processor_model, language)),
        ("Saving Logits", lambda logits: save_logits(logits, output_path)),
        ("Testing CUDA Tensor Bridge", lambda logits: cuda_utils.test_tensor_bridge(logits))
    ]
    
    results = {}
    for i, (step_name, step_func) in enumerate(steps):
        start_time = time.time()
        print(f"\n=== Step {i+1}: {step_name} ===")
        
        # Handle function arguments based on previous results
        if step_name == "Loading Audio":
            results['waveform'] = step_func()
        elif step_name == "Loading Whisper Model":
            results['processor_model'] = step_func()
        elif step_name == "Extracting Logits":
            results['logits'] = step_func(results['waveform'], results['processor_model'])
        elif step_name == "Saving Logits":
            step_func(results['logits'])
        elif step_name == "Testing CUDA Tensor Bridge":
            step_func(results['logits'])
            
        elapsed = time.time() - start_time
        print(f"{step_name} completed in {elapsed:.2f} seconds")
    
    print(f"\nPipeline completed successfully!")
    print(f"Logits saved to: {output_path}")
    
    return output_path

def main():
    """Main entry point, handles command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Process audio through Whisper and CUDA beam search pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Audio input options
    parser.add_argument("audio", nargs="?", 
                        default=AUDIO_DIR / "common_voice_es_41913637.mp3",
                        help="Path to audio file")
    
    # Output options
    parser.add_argument("--output", "-o", help="Path to save logits file")
    
    # Model options
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_SIZE, 
                        choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model size")
    
    # Language options
    parser.add_argument("--language", "-l", default=DEFAULT_LANGUAGE, 
                        help="Language code (e.g., 'en', 'es')")
    
    # Device options
    parser.add_argument("--device", "-d", choices=["cpu", "cuda"], default=None, 
                        help="Device to run on (default: CUDA if available)")
    
    # Pipeline options
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction and load existing logits file")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip extraction if output file already exists")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 60)
    print("CUDA Beam Search Pipeline".center(60))
    print("=" * 60)
    
    # Check CUDA availability
    print("\n=== CUDA Availability ===")
    cuda_utils.check_cuda_availability()
    
    # Process audio or load existing logits
    if args.skip_extraction:
        print("\n=== Loading Existing Logits ===")
        output_path = args.output or LOGITS_DIR / "logits.bin"
        print(f"Loading logits from: {output_path}")
        logits = cuda_utils.load_logits_from_file(output_path)
        print(f"Logits shape: {logits.shape}")
        
        print("\n=== Testing CUDA Tensor Bridge ===")
        cuda_utils.test_tensor_bridge(logits)
    else:
        # Run the full pipeline
        process_audio(
            audio_path=args.audio,
            output_path=args.output,
            model_name=args.model,
            language=args.language,
            device=args.device,
            skip_existing=args.skip_existing
        )
    
    print("\nDone!")

if __name__ == "__main__":
    main() 