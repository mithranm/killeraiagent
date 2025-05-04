"""
Unified import for all LLM backends.

Users can import from killeraiagent.models directly, or from submodules.
"""

from killeraiagent.models.base import (
    LLM,
    EmbeddingModel,
    ModelInfo,
    ChatMessage,
    NullModel
)

from killeraiagent.models.llama_cpp.llama_cpp_cli import LlamaCppCLI
from killeraiagent.models.llama_cpp.llama_cpp_server import LlamaCppServer
from killeraiagent.models.openai import OpenAILLM
from killeraiagent.models.transformers.huggingface_llm import TransformersLLM
from killeraiagent.models.factory import create_llm_instance

__all__ = [
    "LLM",
    "EmbeddingModel",
    "ModelInfo",
    "ChatMessage",
    "LlamaCppCLI",
    "LlamaCppServer",
    "OpenAILLM",
    "TransformersLLM",
    "NullModel",
    "create_llm_instance",
]
