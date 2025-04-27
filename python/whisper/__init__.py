"""
Whisper utilities for extracting and working with logits.
"""
# Import necessary functions from whisper_utils to expose them at the package level
from .whisper_utils import (
    load_audio,
    load_whisper_model,
    extract_logits,
    save_logits
) 