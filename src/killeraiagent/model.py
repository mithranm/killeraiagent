"""
Universal LLM interface for KAIA.

This module provides a unified interface for interacting with LLM backends using either
Hugging Face Transformers or llama-cpp-python. Instead of a global manager, the module
provides a factory function `create_model` that lets developers create LLM instances by supplying
a model path, backend choice, and optional chat template.
"""

import os
import gc
import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Resource management (unchanged)
from killeraiagent.resources import get_resource_manager

logger = logging.getLogger(__name__)


class ModelInfo:
    """Information about an LLM model."""
    
    def __init__(
        self,
        model_id: str,
        model_path: Optional[str] = None,
        model_type: str = "llamacpp",  # "llamacpp" or "hf" (huggingface)
        context_length: int = 4096,
        requires_gpu: bool = False,
        model_size_gb: float = 0.0,
        description: str = "",
        quantization: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.model_type = model_type
        self.context_length = context_length
        self.requires_gpu = requires_gpu
        self.model_size_gb = model_size_gb
        self.description = description
        self.quantization = quantization
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "context_length": self.context_length,
            "requires_gpu": self.requires_gpu,
            "model_size_gb": self.model_size_gb,
            "description": self.description,
            "quantization": self.quantization,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        return cls(
            model_id=data["model_id"],
            model_path=data.get("model_path"),
            model_type=data.get("model_type", "llamacpp"),
            context_length=data.get("context_length", 4096),
            requires_gpu=data.get("requires_gpu", False),
            model_size_gb=data.get("model_size_gb", 0.0),
            description=data.get("description", ""),
            quantization=data.get("quantization"),
        )


class Model:
    """Abstract base class for all LLM implementations."""
    
    def __init__(self, model_info: ModelInfo, **kwargs):
        self.model_info = model_info
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7, 
        top_p: float = 0.9, 
        repeat_penalty: float = 1.1, 
        **kwargs
    ) -> Tuple[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")
    
    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        gc.collect()


class LlamaCppModel(Model):
    """Implementation for llama-cpp-python models."""
    
    def __init__(self, model_info: ModelInfo, **kwargs):
        super().__init__(model_info, **kwargs)
        self.chat_format = kwargs.get("chat_format", "custom")
        self.chat_template = kwargs.get("chat_template", (
            "{%- if messages[0]['role'] == 'system' -%}"
            "<|im_start|>system\n{{ messages[0]['content'] }}<|im_end|>\n"
            "{%- endif -%}"
            "{%- for message in messages[1:] -%}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{%- endfor -%}"
            "<|im_start|>assistant\n"
        ))
    
    def load(self) -> bool:
        if self.is_loaded and self.model is not None:
            return True
        try:
            import llama_cpp
            model_path = self.model_info.model_path
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Model path not found: {model_path}")
                return False
            logger.info(f"Loading llama-cpp model from: {model_path}")
            # Optional: you could obtain hardware-optimized configuration here if desired.
            n_ctx = self.kwargs.get("n_ctx", self.model_info.context_length)
            n_threads = self.kwargs.get("n_threads", 4)
            n_gpu_layers = self.kwargs.get("n_gpu_layers", 0)
            logger.info(f"Loading with context={n_ctx}, threads={n_threads}, gpu_layers={n_gpu_layers}")
            model_kwargs = {
                "model_path": model_path,
                "n_ctx": n_ctx,
                "n_threads": n_threads,
                "verbose": self.kwargs.get("verbose", False),
                "chat_format": self.chat_format,
                "chat_template": self.chat_template
            }
            if n_gpu_layers > 0:
                model_kwargs["n_gpu_layers"] = n_gpu_layers
            self.model = llama_cpp.Llama(**model_kwargs)
            self.is_loaded = True
            logger.info(f"Successfully loaded {os.path.basename(model_path)}")
            return True
        except Exception as e:
            logger.error(f"Error loading llama-cpp model: {e}")
            self.model = None
            self.is_loaded = False
            return False
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7, 
        top_p: float = 0.9, 
        repeat_penalty: float = 1.1, 
        **kwargs
    ) -> Tuple[str, Any]:
        if not self.is_loaded and not self.load():
            return "Error: Model not loaded", None
        try:
            gen_kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "stop": kwargs.get("stop", ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]),
                "echo": False,
                "stream": False,
            }
            for k, v in kwargs.items():
                if k not in gen_kwargs and not k.startswith("_"):
                    gen_kwargs[k] = v
            start_time = time.time()
            completion = self.model(prompt, **gen_kwargs)
            logger.debug(f"Generation took {time.time() - start_time:.2f}s")
            if (isinstance(completion, dict)
                and "choices" in completion
                and len(completion["choices"]) > 0
                and "text" in completion["choices"][0]):
                text = completion["choices"][0]["text"].strip()
                return text, completion
            else:
                logger.warning(f"Unexpected completion format: {completion}")
                return str(completion), completion
        except Exception as e:
            logger.error(f"Error in llamacpp generation: {e}")
            return f"Error: {str(e)}", None
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if not self.is_loaded and not self.load():
            raise RuntimeError("Model not loaded")
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        embeddings = []
        for t in texts:
            try:
                emb = self.model.embed(t)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                raise
        return embeddings[0] if single_input else embeddings


class HuggingFaceModel(Model):
    """Implementation for Hugging Face Transformers models."""
    
    def __init__(self, model_info: ModelInfo, **kwargs):
        super().__init__(model_info, **kwargs)
        self.model_architecture = None
        self.task = None
        self.device = kwargs.get("device", None)
        self.pipeline = None
        self.chat_template = kwargs.get("chat_template", None)  # Optional chat template
    
    def _determine_device(self) -> str:
        if self.device:
            return self.device
        resource_manager = get_resource_manager()
        if resource_manager.hardware.has_cuda:
            return "cuda:0"
        if resource_manager.hardware.has_mps:
            return "mps"
        return "cpu"
    
    def load(self) -> bool:
        if self.is_loaded and self.model is not None:
            return True
        try:
            import torch
            import transformers
            model_id = self.model_info.model_path or self.model_info.model_id
            logger.info(f"Loading Hugging Face model: {model_id}")
            config = transformers.AutoConfig.from_pretrained(model_id)
            if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
                self.model_architecture = "seq2seq"
            else:
                model_type = getattr(config, 'model_type', '').lower()
                if model_type in ('t5', 'bart', 'pegasus', 'marian', 'mt5'):
                    self.model_architecture = "seq2seq"
                else:
                    model_id_lower = model_id.lower()
                    seq2seq_models = ['t5', 'bart', 'pegasus', 'flan-t5', 'marian', 'mt5']
                    if any(name in model_id_lower for name in seq2seq_models):
                        self.model_architecture = "seq2seq"
                    else:
                        self.model_architecture = "causal"
            self.task = "text2text-generation" if self.model_architecture == "seq2seq" else "text-generation"
            device_name = self._determine_device()
            logger.info(f"Using device: {device_name}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            if self.model_architecture == "seq2seq":
                self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)
                logger.info(f"Loaded sequence-to-sequence model: {model_id}")
            else:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
                logger.info(f"Loaded causal language model: {model_id}")
            self.model.to(device_name)
            self.pipeline = transformers.pipeline(
                self.task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_name
            )
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {e}")
            self.is_loaded = False
            return False
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7, 
        top_p: float = 0.9, 
        repeat_penalty: float = 1.1, 
        **kwargs
    ) -> Tuple[str, Any]:
        if not self.is_loaded and not self.load():
            return "Error: Failed to load model", None
        try:
            if self.model_architecture == "seq2seq":
                gen_kwargs = {
                    "max_length": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                if temperature > 0:
                    gen_kwargs["do_sample"] = True
            else:
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                if repeat_penalty > 1.0:
                    gen_kwargs["repetition_penalty"] = repeat_penalty
                if temperature > 0:
                    gen_kwargs["do_sample"] = True
            for k, v in kwargs.items():
                if k not in gen_kwargs and not k.startswith("_"):
                    gen_kwargs[k] = v
            start_time = time.time()
            result = self.pipeline(prompt, **gen_kwargs)
            logger.debug(f"Generation took {time.time() - start_time:.2f}s")
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    text = result[0]["generated_text"]
                    if self.model_architecture == "causal" and text.startswith(prompt):
                        text = text[len(prompt):]
                    return text.strip(), result
                else:
                    logger.warning(f"Unexpected HF pipeline format: {result}")
                    return str(result), result
            else:
                logger.warning(f"Unexpected HF pipeline result type: {type(result)}")
                return str(result), result
        except Exception as e:
            logger.error(f"Error in Hugging Face generation: {e}")
            return f"Error: {str(e)}", None
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if not self.is_loaded and not self.load():
            raise RuntimeError("Model not loaded")
        try:
            import torch
            single_input = isinstance(text, str)
            texts = [text] if single_input else text
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings[0].tolist() if single_input else [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


def create_model(*, 
                 model_path: str, 
                 backend: str = "hf", 
                 chat_template: Optional[str] = None, 
                 context_length: int = 4096, 
                 **kwargs) -> Model:
    """
    Factory function to create an LLM instance.
    
    Arguments:
        model_path: A local file path for a llama-cpp model or a Hugging Face model identifier.
        backend: Which backend to use – "hf" for Hugging Face Transformers or "llamacpp" for llama-cpp-python.
        chat_template: Optional custom chat template for conversational models.
        context_length: Maximum context length (default 4096).
        **kwargs: Additional parameters to pass to the underlying model (e.g., n_threads, n_gpu_layers).
    
    Returns:
        An instance of HuggingFaceModel or LlamaCppModel, based on the backend parameter.
    """
    # Use the basename (or a derived value) as the model_id
    model_id = os.path.basename(model_path)
    backend = backend.lower()
    if backend not in ("hf", "llamacpp"):
        raise ValueError("backend must be either 'hf' (Hugging Face) or 'llamacpp' (llama-cpp-python)")
    
    # Create a ModelInfo instance – developers can override requires_gpu via kwargs if needed.
    model_info = ModelInfo(
        model_id=model_id,
        model_path=model_path,
        model_type=backend,
        context_length=context_length,
        requires_gpu=(backend == "hf"),  # By default, assume hf models require GPU; adjust as needed.
        description=f"LLM created from {model_path}",
        quantization=kwargs.pop("quantization", None)
    )
    
    # If a chat_template is provided, pass it along.
    if backend == "hf":
        return HuggingFaceModel(model_info, chat_template=chat_template, **kwargs)
    else:
        return LlamaCppModel(model_info, chat_template=chat_template, **kwargs)
