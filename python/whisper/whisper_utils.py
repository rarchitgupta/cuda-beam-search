#!/usr/bin/env python3
"""
Utility functions for working with Whisper models and audio processing.
"""
import os
import torch
import torchaudio
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import subprocess
from functools import lru_cache

# Change from relative import to absolute import
from config import AUDIO_DIR, LOGITS_DIR, DEFAULT_SAMPLE_RATE

def load_audio(audio_path, target_sample_rate=DEFAULT_SAMPLE_RATE):
    """
    Load and preprocess audio for Whisper
    
    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate (Whisper expects 16kHz)
        
    Returns:
        Preprocessed waveform
    """
    print(f"Loading audio from: {audio_path}")
    
    # Resolve path - check various locations
    for path in [
        audio_path,
        os.path.join(os.getcwd(), audio_path),
        os.path.join(AUDIO_DIR, os.path.basename(audio_path)),
        os.path.join("python", audio_path)
    ]:
        if os.path.exists(path):
            audio_path = path
            break
    else:
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Try different loading methods
    try:
        # 1. Primary method: torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
    except RuntimeError as e:
        print(f"Warning: torchaudio.load failed, trying alternatives: {e}")
        
        # 2. Try ffmpeg + soundfile
        try:
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav:
                subprocess.run(['ffmpeg', '-i', audio_path, '-ar', str(target_sample_rate), '-ac', '1', '-y', temp_wav.name], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                waveform, sample_rate = sf.read(temp_wav.name)
                waveform = torch.tensor(waveform).unsqueeze(0)
        except (ImportError, subprocess.SubprocessError):
            # 3. Try pydub
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
                raise RuntimeError("Failed to load audio. Please install: pip install soundfile pydub")
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate}Hz to {target_sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    
    return waveform

@lru_cache(maxsize=1)  # Cache the last loaded model
def load_whisper_model(model_name="large-v3", device=None):
    """
    Load Whisper model and processor
    
    Args:
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        device: Device to load model on ('cpu', 'cuda')
    
    Returns:
        Tuple of (processor, model)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    full_model_name = f"openai/whisper-{model_name}"
    print(f"Loading Whisper {model_name} model on {device}")
    
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
        sampling_rate=DEFAULT_SAMPLE_RATE, 
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
        print(f"WARNING: Logits vocab dimension ({logits.shape[-1]}) differs from Whisper vocab size ({vocab_size})")
    
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
    # Ensure output path is in logits directory by default
    if not os.path.isabs(output_path) and not str(output_path).startswith(str(LOGITS_DIR)):
        output_path = LOGITS_DIR / os.path.basename(output_path)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to numpy and save
    logits_np = logits.cpu().numpy()
    logits_np.tofile(output_path)
    file_size = os.path.getsize(output_path)
    
    print(f"Logits exported to '{output_path}' ({file_size/1024:.1f} KB)")
    
    if save_metadata:
        # Create metadata dictionary
        metadata = {
            "shape": list(logits_np.shape),
            "dtype": str(logits_np.dtype),
            "vocab_size": logits_np.shape[-1],
            "min_value": float(logits_np.min()),
            "max_value": float(logits_np.max()),
            "mean_value": float(logits_np.mean()),
            "std_value": float(logits_np.std())
        }
        
        # Save as text
        with open(f"{output_path}.meta", "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        # Save as JSON
        with open(f"{output_path}.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to '{output_path}.meta' and '{output_path}.json'")
    
    if save_sample_values:
        # Print first 10 logits
        print(f"Sample logits (first 10 values):")
        print(logits_np[0, 0, :10]) 