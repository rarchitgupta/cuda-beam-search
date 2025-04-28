#!/usr/bin/env python3
"""
Whisper-specific CUDA beam search decoder implementation.
This provides an easy-to-use interface for integrating CUDA-accelerated beam search
with Hugging Face's Whisper implementation.
"""

import torch
from cuda_beam_search import TensorBridge, BeamSearchConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig


class WhisperBeamSearchDecoder:
    """
    A specialized beam search decoder for Whisper models that uses CUDA acceleration.
    
    This class provides an efficient implementation of beam search for Whisper models,
    replacing the standard PyTorch beam search implementation with a CUDA-accelerated version.
    """
    
    def __init__(self, model, processor=None):
        """
        Initialize the WhisperBeamSearchDecoder.
        
        Args:
            model (WhisperForConditionalGeneration): The Whisper model to use for decoding
            processor (WhisperProcessor, optional): The Whisper processor for input/output processing
        """
        if not isinstance(model, WhisperForConditionalGeneration):
            raise TypeError("Model must be an instance of WhisperForConditionalGeneration")
        
        self.model = model
        self.processor = processor
        self.device = model.device
        self.tensor_bridge = TensorBridge()  # Initialize the CUDA beam search bridge
        
        # Default beam search configuration
        self.beam_config = BeamSearchConfig()
        self.beam_config.beam_width = 5
        self.beam_config.temperature = 1.0
        self.beam_config.top_k = 50
        self.beam_config.top_p = 1.0
        self.beam_config.max_length = 448  # Whisper's default max length
        
        # Get language/task tokens from model
        self.generation_config = model.generation_config
    
    def update_config_from_generation_config(self, generation_config=None):
        """
        Update beam search configuration based on Whisper's generation config.
        
        Args:
            generation_config (GenerationConfig, optional): The generation configuration to use
        """
        if generation_config is None:
            generation_config = self.generation_config
            
        # Map Whisper generation parameters to our beam config
        self.beam_config.beam_width = getattr(generation_config, "num_beams", 5)
        self.beam_config.temperature = getattr(generation_config, "temperature", 1.0)
        self.beam_config.top_k = getattr(generation_config, "top_k", 50)
        self.beam_config.top_p = getattr(generation_config, "top_p", 1.0)
        self.beam_config.max_length = getattr(generation_config, "max_length", 448)
        
        # Extract stop token IDs (e.g., EOS token)
        if hasattr(generation_config, "eos_token_id"):
            self.beam_config.stop_token_ids = [generation_config.eos_token_id]
    
    def decode(self, input_features, generation_config=None, return_timestamps=False, 
               language=None, task=None, return_dict_in_generate=False, **kwargs):
        """
        Decode audio features using CUDA-accelerated beam search.
        
        Args:
            input_features (torch.Tensor): Whisper input features tensor
            generation_config (GenerationConfig, optional): Generation configuration
            return_timestamps (bool): Whether to return timestamps
            language (str, optional): Language to decode (e.g., "en", "fr")
            task (str, optional): Task to perform (e.g., "transcribe", "translate")
            return_dict_in_generate (bool): Whether to return a dict like Whisper's generate method
            **kwargs: Additional arguments passed to the generation config
            
        Returns:
            torch.Tensor: Generated token ids
        """
        # Update configuration if provided
        if generation_config is not None:
            self.update_config_from_generation_config(generation_config)
        
        # Process additional kwargs to update beam config
        for key, value in kwargs.items():
            if hasattr(self.beam_config, key):
                setattr(self.beam_config, key, value)
        
        # Get encoder outputs
        encoder_outputs = self.model.get_encoder()(input_features)
        
        # Initialize decoder inputs with appropriate tokens (language, task tokens)
        decoder_input_ids = self._get_initial_decoder_input_ids(language, task, return_timestamps)
        
        # Start decoding loop
        max_length = self.beam_config.max_length
        batch_size = input_features.shape[0]
        all_decoder_input_ids = decoder_input_ids.repeat(batch_size, 1)
        
        # Get entire prediction in one step for each batch item
        generated_ids = []
        for batch_idx in range(batch_size):
            batch_decoder_ids = all_decoder_input_ids[batch_idx:batch_idx+1]
            batch_encoder_hidden_states = encoder_outputs[0][batch_idx:batch_idx+1]
            
            # Generate sequence for this batch item
            sequence = self._generate_sequence(
                batch_encoder_hidden_states,
                batch_decoder_ids,
                max_length
            )
            
            generated_ids.append(sequence)
        
        # Stack results
        generated_ids = [torch.tensor(ids, device=self.device) for ids in generated_ids]
        max_len = max(ids.size(0) for ids in generated_ids)
        padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        
        for i, ids in enumerate(generated_ids):
            padded_ids[i, :ids.size(0)] = ids
        
        # Return in the format expected by Whisper
        if return_dict_in_generate:
            return {"sequences": padded_ids}
        return padded_ids
    
    def _get_initial_decoder_input_ids(self, language=None, task=None, return_timestamps=False):
        """
        Get the initial decoder input IDs for the generation.
        
        Args:
            language (str, optional): Language to decode
            task (str, optional): Task to perform
            return_timestamps (bool): Whether to return timestamps
            
        Returns:
            torch.Tensor: Initial decoder input IDs
        """
        # Use processor if available, otherwise fall back to model config
        if self.processor is not None:
            # Get tokenizer
            tokenizer = self.processor.tokenizer
            
            # Start with decoder start token
            decoder_input_ids = [self.model.config.decoder_start_token_id]
            
            # Add language token if specified
            if language is not None:
                lang_token_id = tokenizer.convert_tokens_to_ids(f"<|{language}|>")
                decoder_input_ids.append(lang_token_id)
            
            # Add task token if specified
            if task is not None:
                task_token = f"<|{task}|>"
                if return_timestamps:
                    task_token = f"<|{task}|>"
                task_token_id = tokenizer.convert_tokens_to_ids(task_token)
                decoder_input_ids.append(task_token_id)
            
            # Add previous token marker
            prev_token_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            decoder_input_ids.append(prev_token_id)
            
            # Convert to tensor
            decoder_input_ids = torch.tensor([decoder_input_ids], device=self.device)
        else:
            # Fall back to default behavior if processor not available
            decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)
        
        return decoder_input_ids
    
    def _generate_sequence(self, encoder_hidden_states, decoder_input_ids, max_length):
        """
        Generate a sequence using beam search.
        
        Args:
            encoder_hidden_states (torch.Tensor): Encoder hidden states
            decoder_input_ids (torch.Tensor): Initial decoder input IDs
            max_length (int): Maximum sequence length
            
        Returns:
            list: List of token IDs
        """
        # Initialize with provided decoder input ids
        current_ids = decoder_input_ids.clone()
        
        # Initialize sequence with initial decoder input
        final_sequence = current_ids.squeeze(0).tolist()
        
        # Generate tokens auto-regressively
        for i in range(max_length - len(final_sequence)):
            # Get decoder outputs for current position
            decoder_outputs = self.model.decoder(
                input_ids=current_ids,
                encoder_hidden_states=encoder_hidden_states
            )
            
            # Project decoder outputs to vocabulary
            logits = self.model.proj_out(decoder_outputs[0])
            
            # Get logits for the next token (last position)
            next_token_logits = logits[:, -1, :]
            
            # Execute beam search for this step
            self.tensor_bridge.set_logits_tensor(next_token_logits)
            self.tensor_bridge.execute_beam_search(self.beam_config)
            
            # Get beam search results
            token_sequences = self.tensor_bridge.get_beam_search_results()
            
            # Get next token (best beam)
            if token_sequences and len(token_sequences[0]) > 0:
                next_token_id = token_sequences[0][-1]
            else:
                # Fallback in case beam search fails
                next_token_id = next_token_logits.argmax(dim=-1).item()
            
            # Add to sequence
            final_sequence.append(next_token_id)
            
            # Update decoder input for next iteration
            current_ids = torch.cat([
                current_ids, 
                torch.tensor([[next_token_id]], device=self.device)
            ], dim=1)
            
            # Stop if we hit EOS
            if next_token_id == self.model.config.eos_token_id:
                break
        
        return final_sequence 