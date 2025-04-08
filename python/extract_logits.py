#!/usr/bin/env python3
import os
import numpy as np
import whisper
import torch
import argparse

def extract_logits(audio_file, output_file="logits.bin", model_name="base", device=None):
    """
    Extract logits from Whisper model for a given audio file
    
    Args:
        audio_file: Path to the audio file
        output_file: Path to save the logits binary file
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        device: Device to run the model on ('cpu', 'cuda')
    """
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        return
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Whisper model '{model_name}' on {device}...")
    model = whisper.load_model(model_name).to(device)
    
    # Load and preprocess audio
    print(f"Processing audio: {audio_file}")
    audio = whisper.load_audio(audio_file)
    mel = whisper.log_mel_spectrogram(audio).to(device)
    
    # Run encoder
    print("Running Whisper encoder...")
    with torch.no_grad():
        encoder_output = model.encoder(mel.unsqueeze(0))
        
        # Get initial tokens (start of transcription)
        initial_tokens = torch.tensor([[model.tokenizer.sot, model.tokenizer.sot_prev, model.tokenizer.sot_lm]]).to(device)
        
        # Run decoder to get logits for the first step
        print("Running Whisper decoder to get logits...")
        decoder_output = model.decoder(initial_tokens, encoder_output)
        logits = decoder_output.logits[:, -1]
        
    # Save logits to binary file
    logits_np = logits.cpu().numpy()
    print(f"Saving logits with shape {logits_np.shape} to {output_file}")
    logits_np.tofile(output_file)
    
    # Save metadata for reference
    with open(f"{output_file}.meta", "w") as f:
        f.write(f"Shape: {logits_np.shape}\n")
        f.write(f"Dtype: {logits_np.dtype}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Audio: {audio_file}\n")
    
    print(f"Logits extraction complete. Saved to {output_file}")
    return logits_np.shape

def main():
    parser = argparse.ArgumentParser(description="Extract logits from Whisper model")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--output", default="logits.bin", help="Output file path")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model size")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, 
                        help="Device to run on (default: use CUDA if available)")
    
    args = parser.parse_args()
    extract_logits(args.audio, args.output, args.model, args.device)

if __name__ == "__main__":
    main() 