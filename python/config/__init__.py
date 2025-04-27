"""
Shared configuration and constants for the CUDA beam search project.
"""
import os
from pathlib import Path

# Path configuration
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ASSETS_DIR = PROJECT_ROOT / "assets"
AUDIO_DIR = ASSETS_DIR / "audio"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGITS_DIR = OUTPUTS_DIR / "logits"

# Whisper model configuration
DEFAULT_MODEL_SIZE = "large"
DEFAULT_LANGUAGE = "es"  # Spanish as default since we're using Spanish examples
DEFAULT_SAMPLE_RATE = 16000  # Whisper expects 16kHz audio

# CUDA configuration
BUILD_DIR = PROJECT_ROOT / "build"
MODULE_NAME = "cuda_beam_search"

# Ensure directories exist
for dir_path in [ASSETS_DIR, AUDIO_DIR, OUTPUTS_DIR, LOGITS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Function to get the build instructions
def get_build_instructions():
    return (
        f"Build the CUDA module with these commands:\n"
        f"  mkdir -p {BUILD_DIR}\n"
        f"  cd {BUILD_DIR}\n"
        f"  cmake {PROJECT_ROOT}\n"
        f"  make -j\n"
        f"Then set PYTHONPATH to include {BUILD_DIR}"
    ) 