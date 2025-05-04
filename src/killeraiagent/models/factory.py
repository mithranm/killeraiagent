"""
Factory module for creating LLM instances.
"""

import logging
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator

from killeraiagent.models.base import LLM, ModelInfo
from killeraiagent.models.transformers import TransformersLLM  
from killeraiagent.models.llama_cpp import LlamaCppCLI, LlamaCppServer
from killeraiagent.models.openai import OpenAILLM

logger = logging.getLogger(__name__)

# Valid engine types
EngineType = Literal["transformers", "llama.cpp", "llama.cpp.server", "openai"]

class TransformersConfig(BaseModel):
    """Configuration for TransformersLLM."""
    pipeline_type: str = "text-generation"
    model_name_or_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    device: Optional[int] = None
    revision: Optional[str] = None
    chat_format: str = "chatml"
    chat_template: Optional[str] = None

class LlamaCppConfig(BaseModel):
    """Configuration for LlamaCppCLI."""
    n_gpu_layers: int = 0
    n_threads: int = 4
    context_size: int = 4096

class LlamaCppServerConfig(BaseModel):
    """Configuration for LlamaCppServer."""
    chat_format: str = "chatml"
    chat_template: Optional[str] = None
    n_ctx: Optional[int] = None
    n_threads: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    seed: int = -1
    grammar_file: Optional[str] = None
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    n_parallel: int = 1
    auto_start: bool = False

class OpenAIConfig(BaseModel):
    """Configuration for OpenAILLM."""
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    default_model: Optional[str] = None
    request_timeout: int = 120

class LLMConfig(BaseModel):
    """Combined configuration for all LLM types."""
    engine: str
    model_info: ModelInfo
    transformers: TransformersConfig = Field(default_factory=TransformersConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    llama_cpp_server: LlamaCppServerConfig = Field(default_factory=LlamaCppServerConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    
    @validator('engine')
    def validate_engine(cls, v):
        valid_engines = ["transformers", "llama.cpp", "llama.cpp.server", "openai"]
        if v not in valid_engines:
            raise ValueError(f"Invalid engine type: {v}. Must be one of: {valid_engines}")
        return v

def create_llm_instance(
    model_info: ModelInfo,
    engine: Optional[str] = None,
    **kwargs
) -> Optional[LLM]:
    """Create an LLM instance based on engine type and model info.
    
    Args:
        model_info: Model metadata
        engine: Engine type override. If None, uses model_info.model_engine
        **kwargs: Engine-specific parameters
    
    Returns:
        Optional[LLM]: Configured LLM instance or None if creation fails
    """
    # Use model_info.model_engine if no override provided
    engine = engine or model_info.model_engine
    
    # Validate engine type
    valid_engines = ["transformers", "llama.cpp", "llama.cpp.server", "openai"]
    if engine not in valid_engines:
        logger.error(f"Unsupported engine type: {engine}. Must be one of: {valid_engines}")
        return None

    try:
        # Create config with validated engine type
        config = LLMConfig(
            engine=engine,
            model_info=model_info,
            transformers=TransformersConfig(**kwargs),
            llama_cpp=LlamaCppConfig(**kwargs),
            llama_cpp_server=LlamaCppServerConfig(**kwargs),
            openai=OpenAIConfig(**kwargs)
        )

        # Create instance based on engine type
        if config.engine == "transformers":
            return TransformersLLM(model_info=model_info, **config.transformers.dict(exclude_none=True))
        elif config.engine == "llama.cpp":
            return LlamaCppCLI(model_info=model_info, **config.llama_cpp.dict(exclude_none=True))
        elif config.engine == "llama.cpp.server":
            # Make sure LlamaCppServer implements unload() method!
            return LlamaCppServer(model_info=model_info, **config.llama_cpp_server.dict(exclude_none=True))
        elif config.engine == "openai":
            # Make sure OpenAILLM implements unload() method!
            return OpenAILLM(model_info=model_info, **config.openai.dict(exclude_none=True))
        else:
            # This should never happen due to validation, but type checker needs it
            logger.error(f"Unsupported engine type: {config.engine}")
            return None

    except Exception as e:
        logger.error(f"Failed to create LLM instance: {e}")
        return None