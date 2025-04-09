"""
KAIA - Killer AI Agents Core Module

This module provides a simple interface for working with LLM instances.
It serves as a thin wrapper around an LLM (created via create_model) that supplies
methods for loading the model, generating text, and embedding text.
"""

import logging
from typing import Union, List, Tuple

from killeraiagent.model import create_model, Model

logger = logging.getLogger(__name__)


class KAIA:
    """
    Main KAIA class providing an interface to an LLM.
    
    Usage:
        # Create a KAIA instance wrapping a Hugging Face model:
        from killeraiagent.kaia import get_kaia
        kaia = get_kaia(model_path="facebook/opt-350m", backend="hf", chat_template="custom template")
        
        # Load the model, generate text, etc.
        kaia.load_model()
        text, metadata = kaia.generate("Hello!")
        embeddings = kaia.embed("Some text")
    """
    
    def __init__(self, model: Model):
        """
        Initialize a KAIA instance with an underlying model.
        
        Args:
            model: An instance of a Model subclass (either HuggingFaceModel or LlamaCppModel).
        """
        self.model = model

    def load_model(self) -> bool:
        """
        Load the underlying model.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        return self.model.load()

    def generate(self, prompt: str, **kwargs) -> Tuple[str, dict]:
        """
        Generate text using the underlying model.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters to pass to the generate method.
            
        Returns:
            A tuple of the generated text and any additional metadata.
        """
        return self.model.generate(prompt, **kwargs)

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using the underlying model.
        
        Args:
            text: A single string or a list of strings.
            
        Returns:
            Embedding vector(s) as a list of floats or a list of lists.
        """
        return self.model.embed(text)


def get_kaia(model_path: str, backend: str = "hf", **kwargs) -> KAIA:
    """
    Factory function that creates a KAIA instance.
    
    Args:
        model_path: A local file path for a llama-cpp model or a Hugging Face model identifier.
        backend: Which backend to use – "hf" for Hugging Face Transformers or "llamacpp" for llama-cpp-python.
        **kwargs: Additional arguments to be passed to model creation (for example, context_length or chat_template).
        
    Returns:
        A KAIA instance wrapping the created LLM.
        
    Raises:
        ValueError: if an invalid backend is specified.
    """
    model = create_model(model_path=model_path, backend=backend, **kwargs)
    logger.info(f"KAIA instance created with model '{model.model_info.model_id}' using backend '{backend}'")
    return KAIA(model)
