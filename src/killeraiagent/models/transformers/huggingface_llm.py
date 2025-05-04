"""
Integration with Hugging Face transformers for text generation and sequence-to-sequence tasks.

This module defines the TransformersLLM class which implements the LLM interface as defined in
killeraiagent/models/base.py. It supports both "text-generation" for causal language models and
"text2text-generation" for sequence-to-sequence models (e.g., Flan-T5-based models). All parameters
are explicitly typed.
"""

import logging
from typing import Any, Dict, List, Tuple, Optional

try:
    from transformers import pipeline
except ImportError:
    raise ImportError(
        "Please install the 'transformers' library to use TransformersLLM. For example: pip install transformers"
    )

from killeraiagent.models.base import LLM, ModelInfo
from killeraiagent.models.llama_cpp.templates import format_messages

logger = logging.getLogger(__name__)


class TransformersLLM(LLM):
    """
    An LLM implementation using Hugging Face's transformers pipelines.
    
    Supports:
      - "text-generation" for causal language models.
      - "text2text-generation" for sequence-to-sequence models.
    
    All constructor parameters are explicitly typed.
    """
    
    def __init__(
        self,
        model_info: ModelInfo,
        *,
        pipeline_type: str = "text-generation",
        model_name_or_path: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        device: Optional[int] = None,
        revision: Optional[str] = None,
        chat_format: str = "chatml",
        chat_template: Optional[str] = None
    ) -> None:
        """
        Args:
            model_info (ModelInfo): Metadata about the model.
            pipeline_type (str): Generation task; e.g., "text-generation" or "text2text-generation".
            model_name_or_path (Optional[str]): Model identifier or local path. If not provided, falls back to
                model_info.model_id (or model_info.model_path if available).
            tokenizer_name_or_path (Optional[str]): Tokenizer identifier or path.
            device (Optional[int]): GPU device index (e.g., 0 for "cuda:0") or -1 for CPU.
            revision (Optional[str]): Optional revision/branch for the model.
            chat_format (str): Chat format style (e.g., "chatml", "llama2"); used when formatting chat prompts.
            chat_template (Optional[str]): Optional Jinja2 template name or raw template content for chat.
        """
        super().__init__(model_info)
        self.pipeline_type: str = pipeline_type
        self.model_name_or_path: str = (
            model_name_or_path
            if model_name_or_path is not None
            else (str(model_info.model_path) if model_info.model_path is not None else model_info.model_id)
        )
        self.tokenizer_name_or_path: Optional[str] = tokenizer_name_or_path
        self.device: Optional[int] = device
        self.revision: Optional[str] = revision
        self.chat_format: str = chat_format
        self.chat_template: Optional[str] = chat_template
        self._pipeline: Optional[Any] = None

    def load(self) -> bool:
        """Load the transformers pipeline into memory.

        Returns:
            bool: True if the pipeline was loaded successfully.
        """
        if self.is_loaded:
            return True
        try:
            logger.info(f"Loading HF pipeline (task={self.pipeline_type}, model={self.model_name_or_path})")
            self._pipeline = pipeline(
                task=self.pipeline_type,
                model=self.model_name_or_path,
                tokenizer=(self.tokenizer_name_or_path if self.tokenizer_name_or_path is not None else self.model_name_or_path),
                device=self.device,
                revision=self.revision
            )
            self.is_loaded = True
            logger.info("Transformers pipeline loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load HF pipeline: {e}")
            self._pipeline = None
            return False

    def close(self) -> None:
        """Unload the pipeline and mark the model as closed."""
        self.unload()

    def unload(self) -> None:
        """Explicit implementation of the abstract unload method."""
        self._pipeline = None
        self.is_loaded = False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        do_sample: bool = False
    ) -> Tuple[str, Any]:
        """
        Generate text from a prompt using the transformers pipeline.

        Args:
            prompt (str): The input prompt.
            max_tokens (int): Maximum new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p (nucleus) sampling probability.
            repeat_penalty (float): Penalty for token repetition.
            do_sample (bool): Whether to use sampling. When True, temperature and top_p are applied.

        Returns:
            Tuple[str, Any]: The generated text and raw output.
        """
        if not self.is_loaded and not self.load():
            return ("Error: Transformers pipeline not loaded", None)

        gen_args: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repeat_penalty,
            "do_sample": do_sample
        }
        
        logger.debug(f"Generating with prompt: {prompt!r}, args: {gen_args}")
        
        if self._pipeline is None:
            return ("Error: _pipeline is None after load", None)
        
        try:
            outputs = self._pipeline(prompt, **gen_args)
        except Exception as e:
            logger.error(f"Pipeline generation error: {e}")
            return (f"Error: {e}", {"error": str(e)})
        
        if not outputs or not isinstance(outputs, list):
            logger.warning(f"Unexpected pipeline output: {outputs}")
            return ("", {"raw_output": outputs})
        
        best_output = outputs[0]
        gen_text = best_output.get("generated_text", "")
        return (gen_text, {"raw_output": outputs})

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        do_sample: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a chat completion based on a list of messages.

        Args:
            messages (List[Dict[str, str]]): Conversation history containing 'role' and 'content'.
            max_tokens (int): Maximum new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling probability.
            repeat_penalty (float): Penalty for repetition.
            do_sample (bool): Whether to sample (set to True for creative responses).

        Returns:
            Dict[str, Any]: A dict formatted similarly to OpenAI's chat completions.
        """
        if not self.is_loaded and not self.load():
            return {"error": "Transformers pipeline not loaded"}
        
        if self.chat_template:
            prompt_str = format_messages(messages, self.chat_template, is_template_content=False)
        else:
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}\n")
            prompt_str = "".join(prompt_parts) + "Assistant:"
        
        # Use the formatted prompt_str instead of an undefined variable
        text, meta = self.generate(
            prompt_str,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            do_sample=do_sample
        )
        
        return {
            "id": f"chatcmpl-{id(self)}",
            "object": "chat.completion",
            "created": 0,
            "model": self.model_info.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }
            ],
            "usage": meta if isinstance(meta, dict) else {}
        }
