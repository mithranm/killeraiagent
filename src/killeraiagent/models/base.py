"""
Base interfaces and abstract classes for model implementations.
"""

import gc
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from pydantic import BaseModel, validator

# Configure logging
logger = logging.getLogger(__name__)

class ModelInfo(BaseModel):
    """Information about a model."""
    model_id: str
    model_path: Optional[Path] = None
    model_engine: str = "llama_cpp"
    context_length: int = 4096
    requires_gpu: bool = False
    model_size_gb: float = 0.0
    description: str = ""
    quantization: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
    
    @validator('model_path', pre=True)
    def validate_model_path(cls, v):
        """Convert string path to Path object if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v


class LLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, model_info: ModelInfo, **kwargs):
        self.model_info = model_info
        self.kwargs = kwargs
        self.model = None
        self.is_loaded = False
        self.start_time = time.time()
        logger.debug(f"Initialized {self.__class__.__name__} with model_id: {model_info.model_id}")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> Tuple[str, Any]:
        ...

    @abstractmethod
    def load(self) -> bool:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...
    
    def set_model_path(self, path: Union[str, Path]) -> None:
        """
        Set or update the model path.
        
        Args:
            path: New path to the model file
        """
        if isinstance(path, str):
            path = Path(path).expanduser().resolve()
        
        self.model_info.model_path = path
        logger.info(f"Updated model path to: {path}")
        
        # If model was already loaded, we need to reload it
        if self.is_loaded:
            logger.info("Model was already loaded, reloading with new path")
            self.unload()
            self.load()

    def uptime(self) -> float:
        return time.time() - self.start_time


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, model_info: ModelInfo, **kwargs):
        self.model_info = model_info
        self.kwargs = kwargs
        self.model = None
        self.is_loaded = False
        self.start_time = time.time()
        logger.debug(f"Initialized {self.__class__.__name__} with model_id: {model_info.model_id}")

    @abstractmethod
    def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        ...

    @abstractmethod
    def load(self) -> bool:
        ...

    @abstractmethod
    def unload(self) -> None:
        self.model = None
        self.is_loaded = False
        gc.collect()
        logger.debug("Model unloaded and garbage collected")
    
    def set_model_path(self, path: Union[str, Path]) -> None:
        """
        Set or update the model path.
        
        Args:
            path: New path to the model file
        """
        if isinstance(path, str):
            path = Path(path).expanduser().resolve()
        
        self.model_info.model_path = path
        logger.info(f"Updated model path to: {path}")
        
        # If model was already loaded, we need to reload it
        if self.is_loaded:
            logger.info("Model was already loaded, reloading with new path")
            self.unload()
            self.load()

    def uptime(self) -> float:
        return time.time() - self.start_time


class NullModel(LLM):
    """A null model that does nothing."""

    def __init__(self, model_info: ModelInfo = ModelInfo(model_id="None"), **kwargs):
        super().__init__(model_info, **kwargs)
        logger.debug("Initialized NullModel")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                 top_p: float = 0.9, repeat_penalty: float = 1.1, **kwargs) -> Tuple[str, Any]:
        return "", None

    def load(self) -> bool:
        return True

    def unload(self) -> None:
        pass


class ChatMessage:
    """Represents a single message in a chat conversation."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ChatMessage":
        return cls(
            role=data["role"],
            content=data["content"]
        )
