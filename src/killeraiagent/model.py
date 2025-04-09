"""
Universal LLM interface for KAIA.

This module provides a unified interface for interacting with LLM backends using either
Hugging Face Transformers or llama-cpp-python. Instead of a global manager, the module
provides a factory function `create_model` that lets developers create LLM instances by supplying
a model path, backend choice, and optional chat template.
"""

import os
import gc
import time
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Resource management (unchanged)
from killeraiagent.resources import get_resource_manager

class LLMInfo:
    """Information about an LLM model."""
    
    def __init__(
        self,
        model_id: str,
        model_path: Path = None,
        model_engine: str = "llamacpp",  # "llamacpp" or "hf" (huggingface)
        context_length: int = 4096,
        requires_gpu: bool = False,
        model_size_gb: float = 0.0,
        description: str = "",
        quantization: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.model_engine = model_engine
        self.context_length = context_length
        self.requires_gpu = requires_gpu
        self.model_size_gb = model_size_gb
        self.description = description
        self.quantization = quantization
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_engine": self.model_engine,
            "context_length": self.context_length,
            "requires_gpu": self.requires_gpu,
            "model_size_gb": self.model_size_gb,
            "description": self.description,
            "quantization": self.quantization,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMInfo":
        return cls(
            model_id=data["model_id"],
            model_path=data.get("model_path"),
            model_engine=data.get("model_engine", "llamacpp"),
            context_length=data.get("context_length", 4096),
            requires_gpu=data.get("requires_gpu", False),
            model_size_gb=data.get("model_size_gb", 0.0),
            description=data.get("description", ""),
            quantization=data.get("quantization"),
        )

class LLM:
    """Abstract base class for all LLM implementations."""
    
    def __init__(self, model_info: LLMInfo, **kwargs):
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
    
    def load(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")
    
    def unload(self) -> None:
        """Base unload method that clears references and triggers garbage collection."""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        gc.collect()

class LlamaCppLLM(LLM):
    """Implementation for llama-cpp-python models."""
    
    def __init__(self, model_info: LLMInfo, **kwargs):
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
                logging.error(f"LLM path not found: {model_path}")
                return False
            logging.info(f"Loading llama-cpp model from: {model_path}")
            n_ctx = self.kwargs.get("n_ctx", self.model_info.context_length)
            n_threads = self.kwargs.get("n_threads", 4)
            n_gpu_layers = self.kwargs.get("n_gpu_layers", 0)
            logging.info(f"Loading with context={n_ctx}, threads={n_threads}, gpu_layers={n_gpu_layers}")
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
            logging.info(f"Successfully loaded {os.path.basename(model_path)}")
            return True
        except Exception as e:
            logging.error(f"Error loading llama-cpp model: {e}")
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
        iloaded = False
        if not self.is_loaded:
            logging.warning("Model not loaded, automatically loading, will unload when done...")
            iloaded = True
            self.load()
        try:
            # Ensure that the underlying model instance is callable.
            if self.model is None or not callable(self.model):
                logging.error("Model instance is not callable. Ensure that the llama-cpp model is loaded correctly.")
                raise RuntimeError("LLM model is not properly loaded or callable.")
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
            logging.debug(f"Generation took {time.time() - start_time:.2f}s")
            if (isinstance(completion, dict)
                and "choices" in completion
                and len(completion["choices"]) > 0
                and "text" in completion["choices"][0]):
                text = completion["choices"][0]["text"].strip()
                return text, completion
            else:
                logging.warning(f"Unexpected completion format: {completion}")
                if iloaded:
                    self.unload()
                return str(completion), completion
        except Exception as e:
            logging.error(f"Error in llamacpp generation: {e}")
            raise
    
    def unload(self) -> None:
        """Properly unload the llama-cpp model to free memory resources."""
        if self.model is not None:
            try:
                logging.info(f"Unloading llama-cpp model: {self.model_info.model_id}")
                if hasattr(self.model, 'close'):
                    self.model.close()
                    logging.debug("Successfully called close() method on model")
            except Exception as e:
                logging.warning(f"Error while trying to close the model: {e}")
        super().unload()

class HuggingFaceLLM(LLM):
    """Implementation for Hugging Face Transformers models."""
    
    def __init__(self, model_info: LLMInfo, **kwargs):
        super().__init__(model_info, **kwargs)
        self.model_architecture = None
        self.task = None
        self.device = kwargs.get("device", None)
        self.pipeline = None
        self.chat_template = kwargs.get("chat_template", None)
    
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
            import transformers
            model_id = self.model_info.model_path or self.model_info.model_id
            logging.info(f"Loading Hugging Face model: {model_id}")
            config = transformers.AutoConfig.from_pretrained(model_id)
            if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
                self.model_architecture = "seq2seq"
            else:
                model_engine = getattr(config, 'model_engine', '').lower()
                if model_engine in ('t5', 'bart', 'pegasus', 'marian', 'mt5'):
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
            logging.info(f"Using device: {device_name}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            if self.model_architecture == "seq2seq":
                self.model = transformers.AutoLLMForSeq2SeqLM.from_pretrained(model_id)
                logging.info(f"Loaded sequence-to-sequence model: {model_id}")
            else:
                self.model = transformers.AutoLLMForCausalLM.from_pretrained(model_id)
                logging.info(f"Loaded causal language model: {model_id}")
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
            logging.error(f"Error loading Hugging Face model: {e}")
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
            logging.debug(f"Generation took {time.time() - start_time:.2f}s")
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    text = result[0]["generated_text"]
                    if self.model_architecture == "causal" and text.startswith(prompt):
                        text = text[len(prompt):]
                    return text.strip(), result
                else:
                    logging.warning(f"Unexpected HF pipeline format: {result}")
                    return str(result), result
            else:
                logging.warning(f"Unexpected HF pipeline result type: {type(result)}")
                return str(result), result
        except Exception as e:
            logging.error(f"Error in Hugging Face generation: {e}")
            raise

class NullModel(LLM):
    """A null model that does nothing."""
    
    def __init__(self):
        super().__init__(LLMInfo(model_id="null"))
    
    def generate(self, prompt: str, **kwargs) -> Tuple[str, Any]:
        return "Null model: no generation", None
    
    def load(self) -> bool:
        return True
    
    def unload(self) -> None:
        pass

def create_model(*, 
                 model_path: str, 
                 backend: str = "hf", 
                 chat_template: Optional[str] = None, 
                 context_length: int = 4096,
                 requires_gpu: Optional[bool] = None,
                 **kwargs) -> LLM:
    """
    Factory function to create an LLM instance.
    
    Arguments:
        model_path: A local file path for a llama-cpp model or a Hugging Face model identifier.
        backend: Which backend to use – "hf" for Hugging Face Transformers or "llamacpp" for llama-cpp-python.
        chat_template: Optional custom chat template for conversational models.
        context_length: Maximum context length (default 4096).
        **kwargs: Additional parameters to pass to the underlying model (e.g., n_threads, n_gpu_layers).
    
    Returns:
        An instance of HuggingFaceLLM or LlamaCppLLM, based on the backend parameter.
    """
    model_id = os.path.basename(model_path)
    backend = backend.lower()
    if backend not in ("hf", "llamacpp"):
        raise ValueError("backend must be either 'hf' (Hugging Face) or 'llamacpp' (llama-cpp-python)")
    
    model_info = LLMInfo(
        model_id=model_id,
        model_path=model_path,
        model_engine=backend,
        context_length=context_length,
        requires_gpu=requires_gpu,
        description=f"LLM created from {model_path}",
        quantization=kwargs.pop("quantization", None)
    )
    
    if backend == "hf":
        return HuggingFaceLLM(model_info, chat_template=chat_template, **kwargs)
    else:
        return LlamaCppLLM(model_info, chat_template=chat_template, **kwargs)
