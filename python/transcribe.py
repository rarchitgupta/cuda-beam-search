#!/usr/bin/env python3
import os
import time
import whisper
import torch

def main():
    """
    Use Whisper to transcribe the first Spanish audio file
    """
    # Path to audio files
    audio_file = "cuda-beam-search/spanish_audio/clips/common_voice_es_41913637.mp3"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        return
    
    print(f"Loading Whisper model...")
    # Load Whisper model (use 'base' for initial tests)
    model = whisper.load_model("base")
    
    print(f"Transcribing audio file: {audio_file}")
    # Measure transcription time
    start_time = time.time()
    
    # Transcribe audio directly using Whisper
    result = model.transcribe(audio_file, language="es")
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nTranscription completed in {elapsed_time:.2f} seconds")
    print(f"Transcription: {result['text']}")

if __name__ == "__main__":
    main() 