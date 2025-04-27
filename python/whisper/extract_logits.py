#!/usr/bin/env python3
import os
import numpy as np
import torch
import torchaudio
import argparse
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path

# Define project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGITS_DIR = os.path.join(OUTPUTS_DIR, "logits")

def load_audio(audio_path, target_sample_rate=16000):
    """
    Load and preprocess audio for Whisper
    
    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate (Whisper expects 16kHz)
        
    Returns:
        Preprocessed waveform
    """
    print(f"Loading audio from: {audio_path}")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        # Try different path variations
        rel_path = os.path.join(os.getcwd(), audio_path)
        if os.path.exists(rel_path):
            audio_path = rel_path
        else:
            # Try with assets/audio path
            alt_path = os.path.join(AUDIO_DIR, os.path.basename(audio_path))
            if os.path.exists(alt_path):
                audio_path = alt_path
            else:
                # Legacy path with "python/" prefix
                legacy_path = os.path.join("python", audio_path)
                if os.path.exists(legacy_path):
                    audio_path = legacy_path
                else:
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Primary method
        waveform, sample_rate = torchaudio.load(audio_path)
    except RuntimeError as e:
        print(f"Warning: torchaudio.load failed with error: {e}")
        print("Trying alternative loading method...")
        
        # Try alternative loading using ffmpeg if available
        try:
            import subprocess
            import tempfile
            import soundfile as sf
            
            # Convert mp3 to wav using ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav:
                subprocess.run(['ffmpeg', '-i', audio_path, '-ar', str(target_sample_rate), '-ac', '1', '-y', temp_wav.name], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                waveform, sample_rate = sf.read(temp_wav.name)
                waveform = torch.tensor(waveform).unsqueeze(0)  # Convert to tensor and add channel dimension
                
        except (ImportError, subprocess.SubprocessError) as alt_err:
            # If soundfile or ffmpeg is not available, try another approach
            try:
                from pydub import AudioSegment
                import io
                
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
                
                buf = io.BytesIO()
                audio.export(buf, format="wav")
                buf.seek(0)
                
                waveform, sample_rate = torchaudio.load(buf)
                
            except ImportError:
                raise RuntimeError(f"Failed to load audio. Please install additional packages: pip install soundfile pydub") from alt_err
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate}Hz to {target_sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    
    return waveform

def load_whisper_model(model_name="large", device=None):
    """
    Load Whisper model and processor
    
    Args:
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        device: Device to load model on ('cpu', 'cuda')
    
    Returns:
        Tuple of (processor, model)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Construct full model name
    full_model_name = f"openai/whisper-{model_name}"
    print(f"Loading Whisper {model_name} model on {device}")
    
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(full_model_name)
    model = WhisperForConditionalGeneration.from_pretrained(full_model_name).to(device)
    model.eval()
    
    return processor, model

def extract_logits(audio_waveform, processor, model, language="en"):
    """
    Extract logits from Whisper model
    
    Args:
        audio_waveform: Processed audio waveform
        processor: Whisper processor
        model: Whisper model
        language: Language code ('en', 'es', etc.)
    
    Returns:
        Logits tensor
    """
    print("Preparing input features")
    
    # Prepare input features
    input_features = processor(
        audio_waveform.squeeze().numpy(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(model.device)
    
    # Get language token ID
    language_token_str = f"<|{language}|>"
    print(f"Using language token: {language_token_str}")
    language_token = processor.tokenizer.convert_tokens_to_ids(language_token_str)
    decoder_input_ids = torch.tensor([[language_token]]).to(model.device)
    
    print("Running Whisper inference")
    
    # Forward pass to obtain logits
    with torch.no_grad():
        outputs = model(input_features=input_features, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
    
    # Verify shape is compatible with Whisper vocabulary size
    vocab_size = processor.tokenizer.vocab_size
    if logits.shape[-1] != vocab_size:
        print(f"WARNING: Extracted logits vocab dimension ({logits.shape[-1]}) doesn't match expected Whisper vocab size ({vocab_size})")
    
    print(f"Extracted logits with shape: {logits.shape}")
    return logits

def save_logits(logits, output_path, save_metadata=True, save_sample_values=True):
    """
    Save logits to binary file with metadata
    
    Args:
        logits: Logits tensor
        output_path: Path to save binary file
        save_metadata: Whether to save metadata file
        save_sample_values: Whether to print sample values
    """
    # Ensure output path is absolute
    if not os.path.isabs(output_path):
        # If not specified as an absolute path, save to the logits directory
        if not output_path.startswith(LOGITS_DIR):
            output_path = os.path.join(LOGITS_DIR, os.path.basename(output_path))
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy and save
    logits_np = logits.cpu().numpy()
    logits_np.tofile(output_path)
    file_size = os.path.getsize(output_path)
    
    print(f"Logits exported to '{output_path}' ({file_size/1024:.1f} KB)")
    
    if save_metadata:
        # Save metadata
        meta_path = f"{output_path}.meta"
        metadata = {
            "shape": list(logits_np.shape),
            "dtype": str(logits_np.dtype),
            "vocab_size": logits_np.shape[-1],
            "min_value": float(logits_np.min()),
            "max_value": float(logits_np.max()),
            "mean_value": float(logits_np.mean()),
            "std_value": float(logits_np.std())
        }
        
        with open(meta_path, "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        # Also save as JSON for easier parsing
        json_path = f"{output_path}.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to '{meta_path}' and '{json_path}'")
    
    if save_sample_values:
        # Print sample values (first 10 logits)
        batch_idx = 0
        seq_idx = 0
        print(f"Sample logits (batch={batch_idx}, seq={seq_idx}, first 10 values):")
        print(logits_np[batch_idx, seq_idx, :10])

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract logits from Whisper model")
    parser.add_argument("--audio", default=os.path.join(AUDIO_DIR, "common_voice_es_41913637.mp3"), 
                        help="Path to audio file")
    parser.add_argument("--output", default=os.path.join(LOGITS_DIR, "logits.bin"), 
                        help="Output path for logits binary file")
    parser.add_argument("--model", default="large", 
                        choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model size")
    parser.add_argument("--language", default="en", 
                        help="Language code (e.g., 'en', 'es')")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, 
                        help="Device to run on (default: use CUDA if available)")
    
    args = parser.parse_args()
    
    # Print summary of parameters
    print("\n=== Whisper Logits Extraction ===")
    print(f"Audio file: {args.audio}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=================================\n")
    
    # Process
    try:
        # Load audio
        waveform = load_audio(args.audio)
        
        # Load model
        processor, model = load_whisper_model(args.model, args.device)
        
        # Extract logits
        logits = extract_logits(waveform, processor, model, args.language)
        
        # Save logits with metadata
        save_logits(logits, args.output)
        
        print("\nLogits extraction completed successfully!")
        
    except Exception as e:
        print(f"\nError during logits extraction: {e}")
        raise

if __name__ == "__main__":
    main()
