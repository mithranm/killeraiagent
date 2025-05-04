# Snapshot

## Filesystem Tree

```
killeraiagent/
└── src/
    ├── killeraiagent/
    │   ├── core/
    │   │   ├── __init__.py
    │   │   └── model_management.py
    │   ├── models/
    │   │   ├── llama_cpp/
    │   │   │   ├── __init__.py
    │   │   │   ├── grammar.py
    │   │   │   ├── llama_cpp_cli.py
    │   │   │   ├── llama_cpp_server.py
    │   │   │   └── templates.py
    │   │   ├── openai/
    │   │   │   ├── __init__.py
    │   │   │   └── openai_llm.py
    │   │   ├── transformers/
    │   │   │   ├── __init__.py
    │   │   │   └── huggingface_llm.py
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   └── factory.py
    │   ├── setup/
    │   │   ├── __init__.py
    │   │   ├── setup_core.py
    │   │   ├── setup_utils.py
    │   │   └── setup_wizard.py
    │   ├── __init__.py
    │   ├── cli.py
    │   ├── features.py
    │   ├── hardware.py
    │   ├── multiturn_teapot.py
    │   └── paths.py
    └── voice/
        ├── engines/
        │   ├── __init__.py
        │   ├── stt.py
        │   └── tts.py
        ├── __init__.py
        ├── api.py
        ├── io.py
        └── speech_audio.py
```

## File Contents

Files are ordered alphabetically by path.

### File: src/killeraiagent/__init__.py

```python

```

---
### File: src/killeraiagent/cli.py

```python
#!/usr/bin/env python
"""
Command-line interface for KillerAI Agent.
"""

import logging
import argparse

from killeraiagent.models import ModelInfo, create_llm_instance
from killeraiagent.paths import get_data_paths
from killeraiagent.hardware import detect_hardware_capabilities, get_optimal_model_config
from killeraiagent.core import (
    load_model,
    initialize_voice_support,
    handle_chat_session
)

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging for the CLI."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    # Get log directory from data paths
    data_paths = get_data_paths()
    log_file = data_paths.logs / "kaia_cli.log"
    data_paths.logs.mkdir(parents=True, exist_ok=True)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    
def chat_mode(args):
    """
    Start a chat session with the specified model.
    
    Args:
        args: Command-line arguments
    """
    # Load specified model or find a default
    model_info = None
    
    # Detect hardware capabilities
    hw = detect_hardware_capabilities()
    
    if args.model:
        model_info = load_model(args.model)
    
    if not model_info:
        # Try to find a suitable default model
        available_models = list_available_models()

        if not available_models:
            logger.error("No models available. Please download a model first.")
            return
        
        # For simplicity, just use the first model
        model_info = ModelInfo(**available_models[0])
        logger.info(f"Using default model: {model_info.model_id}")
    
    # Get optimal configuration based on hardware and model size
    model_size_gb = model_info.model_size_gb
    optimal_config = get_optimal_model_config(hw, model_size_gb)
    
    # Override with command line args if specified
    n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None else optimal_config["n_gpu_layers"]
    n_threads = args.threads if args.threads is not None else optimal_config["n_threads"]
    context_size = args.context_size if args.context_size is not None else optimal_config["context_size"]
    
    # Initialize model with optimal configuration
    llm = create_llm_instance(
        model_info=model_info,
        use_llama_cpp=args.use_llama_cpp,  # New flag
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        context_size=context_size,
        use_mlock=optimal_config["use_mlock"],
        use_mmap=optimal_config["use_mmap"],
        batch_size=optimal_config["batch_size"]
    )
    
    if not llm:
        logger.error("Failed to create LLM instance")
        return
    
    # Load the model
    logger.info(f"Loading model {model_info.model_id}...")
    if not llm.load():
        logger.error("Failed to load model")
        return
    
    # Set up voice if requested
    voice_output, voice_input = None, None
    if args.voice:
        voice_output, voice_input = initialize_voice_support()
    
    # Start interactive chat
    logger.info("Starting chat session. Type 'exit' or 'quit' to end.")
    print(f"\nKAIA Chat with {model_info.model_id}")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Define callback functions
    def on_message(role, content):
        if role == "assistant":
            print(f"\nAssistant: {content}")
    
    def on_error(error_msg):
        print(f"\nError: {error_msg}")
    
    def on_thinking():
        print("Thinking...", end="", flush=True)
    
    # Run the chat session
    try:
        handle_chat_session(
            llm=llm,
            on_message=on_message,
            on_error=on_error,
            on_thinking=on_thinking,
            system_prompt=args.system_prompt,
            voice_output=voice_output,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            max_tokens=args.max_tokens
        )
    finally:
        # Clean up
        llm.unload()
        print("\nChat session ended.")

def teapot_chat_mode(args):
    """
    Start a chat session with the Teapot model.
    
    Args:
        args: Command-line arguments
    """
    try:
        from killeraiagent.multiturn_teapot import MultiturnTeapot
        
        # Initialize teapot
        teapot = MultiturnTeapot()
        
        # Set up voice if requested
        voice_output, voice_input = None, None
        if args.voice:
            voice_output, voice_input = initialize_voice_support()
        
        print("\nKAIA Teapot Chat")
        print("Type 'exit' or 'quit' to end the session.")
        
        # Handle initial query if provided
        if args.query:
            response = teapot.chat(args.query)
            print(f"\nYou: {args.query}")
            print(f"Teapot: {response}")
            
            if voice_output:
                try:
                    wav, sr = voice_output.generate(response)
                    # TODO: Implement audio playback
                    logger.info("Generated voice output")
                except Exception as e:
                    logger.error(f"Failed to generate speech: {e}")
        
        # Interactive loop
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
                break
            
            if not user_input.strip():
                continue
            
            response = teapot.chat(user_input)
            print(f"Teapot: {response}")
            
            if voice_output:
                try:
                    wav, sr = voice_output.generate(response)
                    # TODO: Implement audio playback
                    logger.info("Generated voice output")
                except Exception as e:
                    logger.error(f"Failed to generate speech: {e}")
        
        # Clean up
        teapot.close()
        print("\nChat session ended.")
        
    except ImportError as e:
        logger.error(f"Failed to import MultiturnTeapot: {e}")
        print("Teapot is not available. Make sure teapotai is installed.")

def main():
    """Main entry point for the KAIA CLI."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="KAIA - Killer AI Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--engine", choices=["all", "llamacpp", "hf"], 
                          default="all", help="Filter models by engine type")
    list_parser.add_argument("--sort", choices=["size", "name"], 
                          default="name", help="Sort models by size or name")
    list_parser.add_argument("--detailed", action="store_true", 
                          help="Show detailed model information")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session with a model")
    chat_parser.add_argument("--model", type=str, help="Model ID to use")
    chat_parser.add_argument("--use-llama-cpp", action="store_true", 
                            help="Use llama.cpp backend instead of transformers")
    chat_parser.add_argument("--n-gpu-layers", type=int, default=0, help="Number of GPU layers")
    chat_parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    chat_parser.add_argument("--context-size", type=int, default=4096, help="Context window size")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    chat_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    chat_parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repeat penalty")
    chat_parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    chat_parser.add_argument("--system-prompt", type=str, help="System prompt")
    chat_parser.add_argument("--voice", action="store_true", help="Enable voice mode")
    
    # Teapot chat command
    teapot_parser = subparsers.add_parser("teapot-chat", help="Start a chat session with Teapot")
    teapot_parser.add_argument("--query", type=str, help="Initial query")
    teapot_parser.add_argument("--voice", action="store_true", help="Enable voice mode")
    
    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Start the GUI")
    gui_parser.add_argument("--voice", action="store_true", help="Enable voice mode")
    
    # Global arguments
    parser.add_argument("--gui", action="store_true", help="Start the GUI")
    parser.add_argument("--voice", action="store_true", help="Enable voice mode")
    
    args = parser.parse_args()
    
    # Handle the case where --gui is passed as a global argument
    if args.gui:
        raise NotImplementedError("GUI mode is not implemented yet.")
    
    if args.command == "list":
        models = list_available_models()
        
        # Filter by engine if specified
        if args.engine != "all":
            models = [m for m in models if m["model_engine"] == args.engine]
            
        # Sort models
        if args.sort == "size":
            models.sort(key=lambda x: x["model_size_gb"], reverse=True)
        else:  # sort by name
            models.sort(key=lambda x: x["model_id"].lower())
            
        print(f"\nFound {len(models)} models:")
        for model in models:
            print(f"\n- {model['model_id']}")
            print(f"  Engine: {model['model_engine']}")
            print(f"  Size: {model['model_size_gb']:.1f} GB")
            
            if args.detailed:
                print(f"  Path: {model['model_path']}")
                print(f"  Description: {model['description']}")
                if model.get("quantization"):
                    print(f"  Quantization: {model['quantization']}")
                if model.get("context_length"):
                    print(f"  Context Length: {model['context_length']}")
                print(f"  Requires GPU: {model.get('requires_gpu', False)}")
    
    elif args.command == "chat":
        chat_mode(args)
    
    elif args.command == "teapot-chat":
        teapot_chat_mode(args)
    else:
        # Default to showing help if no command specified
        parser.print_help()

if __name__ == "__main__":
    main()
```

---
### File: src/killeraiagent/core/__init__.py

```python
"""
Core functionality shared between CLI and GUI interfaces for KAIA.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable

from killeraiagent.models.base import LLM

from .model_management import load_model

__all__ = [
    'load_model'
]

logger = logging.getLogger(__name__)

def initialize_voice_support() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Initialize voice input and output support.
    
    Returns:
        Tuple of (voice_output, voice_input) objects or (None, None) if initialization failed
    """
    voice_output = None
    voice_input = None
    
    try:
        from killeraiagent.speech_audio import KokoroTTS, KrokoSTT
        voice_output = KokoroTTS()
        voice_input = KrokoSTT()
        logger.info("Voice support initialized")
    except ImportError as e:
        logger.error(f"Failed to initialize voice support: {e}")
    
    return voice_output, voice_input
 
def format_chat_prompt(conversation: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    """
    Format a conversation history into a prompt suitable for the model.
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys
        system_prompt: Optional system prompt to include
        
    Returns:
        Formatted prompt string
    """
    if system_prompt:
        system = f"<s>[SYSTEM] {system_prompt} [/SYSTEM]\n\n"
    else:
        system = "<s>"
    
    prompt = system
    for msg in conversation:
        if msg["role"] == "user":
            prompt += f"[HUMAN] {msg['content']} [/HUMAN]\n\n"
        elif msg["role"] == "assistant":
            prompt += f"[AI] {msg['content']} [/AI]\n\n"
    
    prompt += "[AI]"
    return prompt

def no_op_callback(*args, **kwargs):
    """
    A no-operation callback function.
    
    This function does nothing and is used as a default for optional callbacks.
    """
    pass

def handle_chat_session(
    llm: LLM,
    on_message: Callable[[str, str], None],
    on_error: Callable[[str], None],
    on_thinking: Callable[[], None] = no_op_callback,
    system_prompt: Optional[str] = None,
    voice_output: Optional[Any] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    max_tokens: int = 1024
) -> List[Dict[str, str]]:
    """
    Run an interactive chat session with the given LLM.
    
    Args:
        llm: LLM instance to use
        on_message: Callback function for new messages (user, content)
        on_error: Callback function for errors
        on_thinking: Optional callback function for when the model is thinking
        system_prompt: Optional system prompt to use
        voice_output: Optional voice output module
        temperature: Temperature parameter for generation
        top_p: Top-p sampling parameter
        repeat_penalty: Repeat penalty parameter
        max_tokens: Maximum tokens to generate
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    conversation = []
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
                break
            
            if not user_input:
                continue
            
            # Add to conversation history
            conversation.append({"role": "user", "content": user_input})
            
            # Notify callback
            on_message("user", user_input)
            
            # Prepare prompt
            prompt = format_chat_prompt(conversation, system_prompt)
            
            # Notify thinking callback if provided
            if on_thinking:
                on_thinking()
            
            # Generate response
            try:
                response, _ = llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty
                )
                
                # Add to conversation history
                conversation.append({"role": "assistant", "content": response})
                
                # Notify callback
                on_message("assistant", response)
                
                # Voice output if enabled
                if voice_output:
                    try:
                        wav, sr = voice_output.generate(response)
                        # TODO: Implement audio playback
                        logger.info("Generated voice output")
                    except Exception as e:
                        logger.error(f"Failed to generate speech: {e}")
            
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                on_error(error_msg)
    
    except KeyboardInterrupt:
        logger.info("Chat session interrupted")
    
    return conversation
```

---
### File: src/killeraiagent/core/model_management.py

```python
"""
Model management functionality including discovery, loading and downloading of models.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from killeraiagent.models.base import ModelInfo
from killeraiagent.paths import get_data_paths

logger = logging.getLogger(__name__)

def get_model_search_paths() -> List[Path]:
    """
    Get all directories where models might be stored, respecting FOSS principles.
    
    This function checks multiple locations where models might be stored:
    1. User-specified paths via KAIA_MODEL_PATHS environment variable
    2. Standard ML model locations (~/.cache/huggingface, ~/.local/share/models)
    3. KAIA's own model directory
    
    Returns:
        List of Path objects to search for models
    """
    search_paths = []
    
    # 1. Check environment variable for user-specified paths
    env_paths = os.environ.get("KAIA_MODEL_PATHS", "")
    if env_paths:
        for path_str in env_paths.split(os.pathsep):
            path = Path(path_str).expanduser().resolve()
            if path.exists() and path.is_dir():
                search_paths.append(path)
    
    # 2. Check standard ML model locations
    standard_paths = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".local" / "share" / "models",
        Path.home() / ".ollama" / "models",
        Path("/usr/local/share/models"),
        Path("/opt/models"),
    ]
    
    for path in standard_paths:
        if path.exists() and path.is_dir():
            search_paths.append(path)
            logger.info(f"Added standard model path: {path}")
    
    # 3. Add KAIA's model directory
    data_paths = get_data_paths()
    search_paths.append(data_paths.models)
    logger.info(f"Added KAIA model path: {data_paths.models}")
    
    return search_paths

def list_available_models() -> List[Dict[str, Any]]:
    """
    List all available models from all configured search paths.
    
    Returns:
        List of model info dictionaries
    """
    models = []
    search_paths = get_model_search_paths()
    seen_model_ids = set()
    
    for base_path in search_paths:
        try:
            if not base_path.exists():
                continue
            
            # Look for model info files (JSON)
            for info_file in base_path.glob("**/*.json"):
                try:
                    with open(info_file, 'r') as f:
                        model_info = json.load(f)
                        if "model_id" in model_info and model_info["model_id"] not in seen_model_ids:
                            models.append(model_info)
                            seen_model_ids.add(model_info["model_id"])
                except Exception as e:
                    logger.warning(f"Failed to load model info from {info_file}: {e}")

            # Look for model files (GGUF, safetensors, bin)
            model_extensions = [".gguf", ".safetensors", ".bin"]
            for ext in model_extensions:
                for model_file in base_path.glob(f"**/*{ext}"):
                    model_id = f"model_{model_file.stem}"
                    if model_id not in seen_model_ids:
                        models.append({
                            "model_id": model_id,
                            "model_path": str(model_file),
                            "model_type": ext.lstrip('.'),
                            "description": f"Found {ext} model at {model_file}"
                        })
                        seen_model_ids.add(model_id)
                        logger.info(f"Found model file '{model_file.name}' at {model_file}")

        except Exception as e:
            logger.error(f"Error scanning {base_path} for models: {e}")
    
    return models

def get_huggingface_token() -> Optional[str]:
    """
    Get HuggingFace token from environment variables or HF CLI.
    
    Returns:
        HuggingFace token if available, None otherwise
    """
    # First check environment variable
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    
    # Then try to get from huggingface-cli
    try:
        from huggingface_hub.hf_api import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except ImportError:
        logger.debug("Could not import huggingface_hub.hf_api.HfFolder")
    except Exception as e:
        logger.debug(f"Error getting HuggingFace token from CLI: {e}")
    
    return None

def load_model(model_id: str) -> Optional[ModelInfo]:
    """
    Load a specific model by ID. If model is not found locally and has HF format,
    offer to download it with user confirmation.
    
    Args:
        model_id: ID of the model to load
        
    Returns:
        ModelInfo object if found, None otherwise
    """
    models = list_available_models()
    
    # First try exact match
    for model_info in models:
        if model_info["model_id"] == model_id:
            return ModelInfo(**model_info)
    
    # Try case-insensitive match
    for model_info in models:
        if model_info["model_id"].lower() == model_id.lower():
            return ModelInfo(**model_info)
    
    # Try partial match
    for model_info in models:
        if model_id.lower() in model_info["model_id"].lower():
            return ModelInfo(**model_info)
    
    # If we get here, model wasn't found locally
    if "/" in model_id and not model_id.startswith("/") and not model_id.endswith("/"):
        try:
            import huggingface_hub
            from huggingface_hub import HfApi
            
            # Get HuggingFace token
            token = get_huggingface_token()
            api = HfApi(token=token)
            
            try:
                # Get model info
                model_info = api.model_info(model_id)
                
                # Calculate model size
                size_bytes = 0
                # Use sibling files to calculate total size
                siblings = getattr(model_info, "siblings", None) or []
                for sibling in siblings:
                    if hasattr(sibling, "size"):
                        size_bytes += sibling.size if sibling.size else 0
                
                size_gb = size_bytes / (1024 * 1024 * 1024)
                
                # Ask for confirmation
                response = input(
                    f"\nModel '{model_id}' not found locally but available on HuggingFace.\n"
                    f"Size: {size_gb:.2f} GB\n"
                    f"Download now? (y/n): "
                )
                
                if response.lower() in ('y', 'yes'):
                    logger.info(f"Downloading model {model_id} from HuggingFace...")
                    
                    # Prepare download location
                    paths = get_data_paths()
                    model_dir = paths.models / model_id.replace("/", "__")
                    model_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Download model
                    local_path = huggingface_hub.snapshot_download(
                        repo_id=model_id,
                        token=token,
                        cache_dir=str(model_dir),
                        local_dir=str(model_dir),
                        local_dir_use_symlinks=False
                    )
                    
                    # Create model info
                    new_model_info = ModelInfo(
                        model_id=model_id,
                        model_path=Path(local_path),
                        model_engine="transformers",
                        model_size_gb=size_gb,
                        description=f"Downloaded from HuggingFace: {model_id}",
                    )
                    
                    logger.info(f"Model downloaded to: {local_path}")
                    return new_model_info
                else:
                    logger.info("Download cancelled by user")
            
            except Exception as e:
                logger.error(f"Error getting model info from HuggingFace: {e}")
                
                # If authorization error, suggest setting HF_TOKEN
                if "401" in str(e) or "403" in str(e) or "unauthorized" in str(e).lower():
                    logger.error("Authentication error. Set the HF_TOKEN environment variable or login with 'huggingface-cli login'")
        
        except ImportError:
            logger.error("huggingface_hub library is required to download models. Please install it with 'pip install huggingface_hub'")
    
    logger.error(f"Model '{model_id}' not found")
    return None
```

---
### File: src/killeraiagent/features.py

```python
"""
Feature detection for optional dependencies.
This module provides a single source of truth for which optional features are available.
"""
import importlib.util
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

class Features:
    """
    Feature detection class to track which optional dependencies are available.
    """
    _instance = None  # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Features, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize features by checking for required dependencies."""
        self._features: Dict[str, bool] = {}
        
        # Voice support
        self._features["voice"] = self._check_feature(
            name="voice",
            modules=["kokoro", "sounddevice", "soundfile"],
        )
        
        # STT support
        self._features["stt"] = self._check_feature(
            name="stt",
            modules=["sherpa_onnx", "torchaudio"],
        )
        
        # GUI support
        self._features["gui"] = self._check_feature(
            name="gui", 
            modules=["PySide6"],
        )
        
        # Log detected features
        available = [f for f, enabled in self._features.items() if enabled]
        logger.info(f"Detected features: {', '.join(available) or 'none'}")
    
    def _check_feature(self, name: str, modules: list[str]) -> bool:
        """Check if all required modules for a feature are available."""
        for module in modules:
            if not importlib.util.find_spec(module):
                logger.debug(f"Feature '{name}' not available: module '{module}' not found")
                return False
        return True
    
    def has(self, feature: str) -> bool:
        """Check if a specific feature is available."""
        return self._features.get(feature, False)
    
    def require(self, feature: str) -> bool:
        """
        Check if a feature is available and log installation instructions if not.
        Returns True if the feature is available, False otherwise.
        """
        if self.has(feature):
            return True
        
        # Provide helpful installation instructions
        logger.warning(f"Feature '{feature}' is not available.")
        logger.warning(f"To enable this feature, install with: pip install killeraiagent[{feature}]")
        return False
    
    @property
    def available(self) -> Set[str]:
        """Get the set of available features."""
        return {f for f, enabled in self._features.items() if enabled}

# Global singleton instance
features = Features()
```

---
### File: src/killeraiagent/hardware.py

```python
"""
Modern hardware detection system for AI acceleration.

This module provides comprehensive detection of hardware capabilities for optimal
AI model inference across CPU, GPU, and specialized accelerators.
"""

import os
import platform
import subprocess
import logging
import shutil
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

import psutil
import torch
from pydantic import BaseModel, Field, validator

from killeraiagent.paths import DataPaths

# Conditionally import winreg with ignores for Pyright
if platform.system() == "Windows":
    import winreg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AcceleratorType(str, Enum):
    """Supported hardware acceleration backends."""
    CPU = "cpu"
    CUDA = "cuda"  # NVIDIA CUDA
    ROCM = "rocm"  # AMD ROCm/HIP
    METAL = "metal"  # Apple Metal
    VULKAN = "vulkan"  # Vulkan (cross-platform)
    OPENCL = "opencl"  # OpenCL (cross-platform)
    SYCL = "sycl"  # Intel SYCL
    MUSA = "musa"  # Moore Threads MUSA
    CANN = "cann"  # Huawei Ascend CANN
    KLEIDIAI = "kleidiai"  # Arm KleidiAI

class CUDAInfo(BaseModel):
    """Information about NVIDIA CUDA installation."""
    available: bool = False
    version: Optional[str] = None
    compute_capability: Optional[str] = None
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)
    driver_version: Optional[str] = None
    nvcc_path: Optional[Path] = None
    cuda_path: Optional[Path] = None
    
    @property
    def cuda_version_major(self) -> Optional[int]:
        """Get major CUDA version number."""
        if not self.version:
            return None
        try:
            return int(self.version.split('.')[0])
        except (ValueError, IndexError):
            return None
    
    @property
    def cuda_version_minor(self) -> Optional[int]:
        """Get minor CUDA version number."""
        if not self.version:
            return None
        try:
            return int(self.version.split('.')[1])
        except (ValueError, IndexError):
            return None
    
    @property
    def cuda_wheel_version(self) -> Optional[str]:
        """Get the PyTorch CUDA wheel version string."""
        if not self.version:
            return None
        
        major = self.cuda_version_major
        minor = self.cuda_version_minor
        
        if major is None or minor is None:
            return None
            
        if major >= 12:
            return "cu121"  # Latest
        elif major == 11:
            if minor >= 8:
                return "cu118"
            elif minor >= 7:
                return "cu117"
            else:
                return "cu116"
        return None


class ROCmInfo(BaseModel):
    """Information about AMD ROCm/HIP installation."""
    available: bool = False
    version: Optional[str] = None
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)
    driver_version: Optional[str] = None
    rocm_path: Optional[Path] = None
    architectures: List[str] = Field(default_factory=list)


class MetalInfo(BaseModel):
    """Information about Apple Metal GPU."""
    available: bool = False
    is_apple_silicon: bool = False
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)


class VulkanInfo(BaseModel):
    """Information about Vulkan support."""
    available: bool = False
    version: Optional[str] = None
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)
    icd_loader_available: bool = False


class OpenCLInfo(BaseModel):
    """Information about OpenCL support."""
    available: bool = False
    version: Optional[str] = None
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)
    platforms: List[Dict[str, Any]] = Field(default_factory=list)


class CPUInfo(BaseModel):
    """Detailed CPU information."""
    count: int = Field(gt=0)
    physical_cores: int = Field(gt=0)
    logical_cores: int = Field(gt=0)
    architecture: str
    model_name: str
    frequency_mhz: float
    supports_avx: bool = False
    supports_avx2: bool = False
    supports_avx512: bool = False
    supports_vnni: bool = False
    supports_arm_sve: bool = False
    supports_arm_neon: bool = False
    supports_kleidiai: bool = False


class MemoryInfo(BaseModel):
    """System memory information."""
    total_gb: float = Field(gt=0)
    available_gb: float = Field(ge=0)
    used_gb: float
    percent_used: float = Field(ge=0, le=100)


class GPUInfo(BaseModel):
    """Unified GPU information across different backends."""
    acceleration_type: AcceleratorType
    name: str
    vendor: str
    driver_version: Optional[str] = None
    total_memory_mb: float = 0
    free_memory_mb: float = 0
    compute_capability: Optional[str] = None
    device_id: int = 0
    pci_bus_id: Optional[str] = None


class LlamaCppConfig(BaseModel):
    """Configuration for llama.cpp."""
    build_type: str = "Release"
    n_gpu_layers: int = 0
    n_threads: int = 4
    acceleration: AcceleratorType = AcceleratorType.CPU
    main_gpu: int = 0
    tensor_split: Optional[List[float]] = None
    executable_path: Optional[Path] = None
    cmake_args: List[str] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    
    @validator('executable_path')
    def check_executable_exists(cls, v):
        """Verify the specified executable exists."""
        if v is not None and not v.exists():
            raise ValueError(f"Executable not found: {v}")
        return v


class HardwareCapabilities(BaseModel):
    """Complete hardware capabilities of the system."""
    cpu: CPUInfo
    memory: MemoryInfo
    primary_acceleration: AcceleratorType = AcceleratorType.CPU
    gpus: List[GPUInfo] = Field(default_factory=list)
    cuda: CUDAInfo = Field(default_factory=CUDAInfo)
    rocm: ROCmInfo = Field(default_factory=ROCmInfo)
    metal: MetalInfo = Field(default_factory=MetalInfo)
    vulkan: VulkanInfo = Field(default_factory=VulkanInfo)
    opencl: OpenCLInfo = Field(default_factory=OpenCLInfo)
    
    @property
    def available_accelerators(self) -> Set[AcceleratorType]:
        """Get the set of available acceleration backends."""
        result = {AcceleratorType.CPU}
        
        if self.cuda.available:
            result.add(AcceleratorType.CUDA)
        
        if self.rocm.available:
            result.add(AcceleratorType.ROCM)
            
        if self.metal.available:
            result.add(AcceleratorType.METAL)
            
        if self.vulkan.available:
            result.add(AcceleratorType.VULKAN)
            
        if self.opencl.available:
            result.add(AcceleratorType.OPENCL)
            
        if self.cpu.supports_kleidiai:
            result.add(AcceleratorType.KLEIDIAI)
            
        return result


def detect_cpu_capabilities() -> CPUInfo:
    """
    Detect CPU capabilities including architecture, cores, and supported instruction sets.
    
    Returns:
        CPUInfo: Detailed CPU information
    """
    cpu_count = os.cpu_count() or 4
    physical_cores = psutil.cpu_count(logical=False) or 2
    logical_cores = psutil.cpu_count(logical=True) or 4
    
    # Get CPU model information
    if platform.system() == "Windows":
        try:
            # Use winreg if available
            if 'winreg' in globals():
                key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as _key:  # type: ignore[attr-defined]
                    model_name = winreg.QueryValueEx(_key, "ProcessorNameString")[0]  # type: ignore[attr-defined]
            else:
                model_name = platform.processor()
        except Exception:
            model_name = platform.processor()
    elif platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], 
                capture_output=True, text=True, check=True
            )
            model_name = result.stdout.strip()
        except Exception:
            model_name = platform.processor()
    else:  # Linux and others
        try:
            with open("/proc/cpuinfo", "r") as f:
                model_name = platform.processor()
                for line in f:
                    if "model name" in line:
                        model_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            model_name = platform.processor()
    
    # Get CPU frequency
    try:
        frequency = psutil.cpu_freq().max
        if not frequency:
            frequency = psutil.cpu_freq().current
    except Exception:
        frequency = 0
    
    # Detect architecture
    arch = platform.machine().lower()
    
    # Detect instruction set support
    supports_avx = False
    supports_avx2 = False
    supports_avx512 = False
    supports_vnni = False
    supports_arm_sve = False
    supports_arm_neon = False
    supports_kleidiai = False
    
    if arch in ("x86_64", "amd64", "i386", "i686"):
        try:
            # Try using cpuinfo library if available
            try:
                import py_cpuinfo  # type: ignore[import]
                info = py_cpuinfo.get_cpu_info()
                flags = info.get("flags", [])
                supports_avx = "avx" in flags
                supports_avx2 = "avx2" in flags
                supports_avx512 = any(flag.startswith("avx512") for flag in flags)
                supports_vnni = "avx512_vnni" in flags or "avx_vnni" in flags
            except ImportError:
                logger.debug("py_cpuinfo not available, using fallback detection")
                # Fallback to manual detection
                if platform.system() == "Linux":
                    try:
                        with open("/proc/cpuinfo", "r") as f:
                            content = f.read().lower()
                            supports_avx = " avx " in content
                            supports_avx2 = " avx2 " in content
                            supports_avx512 = " avx512" in content
                            supports_vnni = (" avx512_vnni" in content 
                                             or " avx_vnni" in content)
                    except Exception:
                        pass
                elif platform.system() == "Darwin":
                    try:
                        result = subprocess.run(
                            ["sysctl", "-n", "machdep.cpu.features"], 
                            capture_output=True, text=True, check=True
                        )
                        features = result.stdout.lower()
                        supports_avx = "avx" in features
                        supports_avx2 = "avx2" in features
                        supports_avx512 = "avx512" in features
                    except Exception:
                        pass
                elif platform.system() == "Windows":
                    # Minimal fallback for Windows if py_cpuinfo isn't installed
                    pass
        except Exception as e:
            logger.warning(f"Failed to detect CPU instruction set support: {e}")
    
    elif arch in ("arm64", "aarch64", "armv8"):
        supports_arm_neon = True  # Most ARM64 processors support NEON
        
        # Check for SVE support
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read().lower()
                    supports_arm_sve = "sve" in content
            except Exception:
                pass
            
        # On Apple Silicon, we know NEON is available
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            supports_arm_neon = True
            
        # Check for ARM KleidiAI support (simplified assumption)
        try:
            result = subprocess.run(["uname", "-a"], capture_output=True, text=True, check=False)
            supports_kleidiai = supports_arm_neon  # Simplified placeholder
        except Exception:
            pass
    
    return CPUInfo(
        count=cpu_count,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        architecture=arch,
        model_name=model_name,
        frequency_mhz=frequency,
        supports_avx=supports_avx,
        supports_avx2=supports_avx2,
        supports_avx512=supports_avx512,
        supports_vnni=supports_vnni,
        supports_arm_sve=supports_arm_sve,
        supports_arm_neon=supports_arm_neon,
        supports_kleidiai=supports_kleidiai
    )


def detect_memory() -> MemoryInfo:
    """
    Detect system memory information.
    
    Returns:
        MemoryInfo: System memory details
    """
    mem = psutil.virtual_memory()
    return MemoryInfo(
        total_gb=mem.total / (1024**3),
        available_gb=mem.available / (1024**3),
        used_gb=(mem.total - mem.available) / (1024**3),
        percent_used=mem.percent
    )


def detect_cuda() -> CUDAInfo:
    """
    Detect NVIDIA CUDA capabilities.
    
    Returns:
        CUDAInfo: CUDA hardware and software information
    """
    result = CUDAInfo(available=False)
    
    # Check if CUDA is available through PyTorch
    if not torch.cuda.is_available():
        return result
    
    # CUDA is available
    result.available = True
    result.device_count = torch.cuda.device_count()
    
    # Get CUDA version
    try:
        # Access CUDA version through PyTorch
        result.version = torch.version.cuda  # type: ignore[attr-defined]
    except Exception:
        # Fallback: parse with nvcc
        try:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                result.nvcc_path = Path(nvcc_path)
                output = subprocess.run(
                    [nvcc_path, "--version"], 
                    capture_output=True, text=True, check=True
                )
                import re
                version_match = re.search(r"release (\d+\.\d+)", output.stdout)
                if version_match:
                    result.version = version_match.group(1)
        except Exception as e:
            logger.warning(f"Failed to detect CUDA version: {e}")
    
    # Get CUDA driver version
    try:
        # Access driver version
        result.driver_version = torch.version.cuda  # type: ignore[attr-defined]
    except Exception:
        # Try nvidia-smi
        try:
            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi:
                output = subprocess.run(
                    [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"], 
                    capture_output=True, text=True, check=True
                )
                result.driver_version = output.stdout.strip()
        except Exception as e:
            logger.warning(f"Failed to detect NVIDIA driver version: {e}")
    
    # Get CUDA path
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        result.cuda_path = Path(cuda_path)
    else:
        # Try to find CUDA path from nvcc
        if result.nvcc_path:
            # nvcc is typically in CUDA_PATH/bin/nvcc
            result.cuda_path = result.nvcc_path.parent.parent
    
    # Get device information
    for i in range(result.device_count):
        try:
            device_props = torch.cuda.get_device_properties(i)
            
            # Get compute capability
            if i == 0:
                result.compute_capability = f"{device_props.major}.{device_props.minor}"
            
            # Calculate memory
            total_memory = device_props.total_memory
            try:
                free_memory = (
                    torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                )
            except Exception:
                free_memory = total_memory * 0.8  # Estimate
                
            device_info = {
                "name": device_props.name,
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "total_memory_mb": total_memory / (1024**2),
                "free_memory_mb": free_memory / (1024**2),
                "multi_processor_count": device_props.multi_processor_count,
                "device_id": i
            }
            result.devices.append(device_info)
        except Exception as e:
            logger.warning(f"Error getting properties for CUDA device {i}: {e}")
            
    return result


def detect_rocm() -> ROCmInfo:
    """
    Detect AMD ROCm/HIP capabilities.
    
    Returns:
        ROCmInfo: ROCm hardware and software information
    """
    result = ROCmInfo(available=False)
    
    # Check if rocm-smi is available
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        return result
    
    # Try to get ROCm version
    try:
        output = subprocess.run(
            [rocm_smi, "--showversion"], 
            capture_output=True, text=True, check=False
        )
        if output.returncode == 0:
            result.available = True
            import re
            version_match = re.search(r"ROCm-SMI Version: ([\d\.]+)", output.stdout)
            if version_match:
                result.version = version_match.group(1)
    except Exception as e:
        logger.warning(f"Failed to detect ROCm version: {e}")
        return result
    
    # Try to get GPU information
    try:
        output = subprocess.run(
            [rocm_smi, "--showallinfo"], 
            capture_output=True, text=True, check=False
        )
        if output.returncode == 0:
            import re
            gpu_matches = re.findall(r"GPU\[(\d+)\]", output.stdout)
            unique_gpus = set(gpu_matches)
            result.device_count = len(unique_gpus)
            
            for gpu_id in unique_gpus:
                mem_total_match = re.search(
                    rf"GPU\[{gpu_id}\].*?Total Memory.*?: (\d+) (\w+)", 
                    output.stdout, re.DOTALL
                )
                mem_used_match = re.search(
                    rf"GPU\[{gpu_id}\].*?Used Memory.*?: (\d+) (\w+)", 
                    output.stdout, re.DOTALL
                )
                
                total_memory_mb = 0
                free_memory_mb = 0
                
                if mem_total_match and mem_used_match:
                    total_val = int(mem_total_match.group(1))
                    total_unit = mem_total_match.group(2)
                    used_val = int(mem_used_match.group(1))
                    used_unit = mem_used_match.group(2)
                    
                    unit_multipliers = {"MB": 1, "GB": 1024, "KB": 1/1024}
                    total_memory_mb = total_val * unit_multipliers.get(total_unit, 1)
                    used_memory_mb = used_val * unit_multipliers.get(used_unit, 1)
                    free_memory_mb = total_memory_mb - used_memory_mb
                
                name_match = re.search(
                    rf"GPU\[{gpu_id}\].*?Card series.*?: (.*?)$", 
                    output.stdout, re.MULTILINE
                )
                name = name_match.group(1) if name_match else f"AMD GPU {gpu_id}"
                
                arch_match = re.search(
                    rf"GPU\[{gpu_id}\].*?gfx\s*:\s*(gfx\d+)", 
                    output.stdout, re.DOTALL
                )
                arch = arch_match.group(1) if arch_match else None
                
                if arch and arch not in result.architectures:
                    result.architectures.append(arch)
                
                device_info = {
                    "name": name,
                    "device_id": int(gpu_id),
                    "total_memory_mb": total_memory_mb,
                    "free_memory_mb": free_memory_mb,
                    "architecture": arch
                }
                result.devices.append(device_info)
    except Exception as e:
        logger.warning(f"Failed to detect ROCm GPU information: {e}")
    
    # Try to get ROCm path
    result.rocm_path = Path("/opt/rocm") if os.path.exists("/opt/rocm") else None
    
    return result


def detect_metal() -> MetalInfo:
    """
    Detect Apple Metal GPU capabilities.
    
    Returns:
        MetalInfo: Metal hardware information
    """
    result = MetalInfo(available=False)
    
    # Only check on macOS
    if platform.system() != "Darwin":
        return result
    
    # Check if we're on Apple Silicon
    result.is_apple_silicon = platform.machine() == "arm64"
    
    # Check if Metal is available through PyTorch
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    result.available = has_mps
    
    if not result.available:
        # Try checking with system_profiler
        try:
            output = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], 
                capture_output=True, text=True, check=True
            )
            result.available = "Metal" in output.stdout
        except Exception:
            pass
    
    # For Apple Silicon, we know there's typically one integrated GPU
    if result.is_apple_silicon and result.available:
        result.device_count = 1
        
        try:
            output = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], 
                capture_output=True, text=True, check=True
            )
            import re
            chip_match = re.search(r"Chipset Model: (.*?)$", output.stdout, re.MULTILINE)
            gpu_name = chip_match.group(1).strip() if chip_match else "Apple GPU"
            
            vram_match = re.search(r"VRAM \(.*?\): (\d+) (\w+)", output.stdout)
            total_memory_mb = 0
            if vram_match:
                vram_val = int(vram_match.group(1))
                vram_unit = vram_match.group(2)
                if vram_unit.upper() == "GB":
                    total_memory_mb = vram_val * 1024
                elif vram_unit.upper() == "MB":
                    total_memory_mb = vram_val
            else:
                system_memory = psutil.virtual_memory().total / (1024**3)
                total_memory_mb = int(system_memory * 0.4 * 1024)
            
            device_info = {
                "name": gpu_name,
                "device_id": 0,
                "total_memory_mb": total_memory_mb,
                "free_memory_mb": total_memory_mb * 0.8,
                "metal_family": "Apple"
            }
            result.devices.append(device_info)
        except Exception as e:
            logger.warning(f"Failed to detect Metal GPU information: {e}")
            # Provide a fallback device
            device_info = {
                "name": "Apple GPU",
                "device_id": 0,
                "total_memory_mb": 4096,
                "free_memory_mb": 3276,
                "metal_family": "Apple"
            }
            result.devices.append(device_info)
    
    return result


def detect_vulkan() -> VulkanInfo:
    """
    Detect Vulkan support.
    
    Returns:
        VulkanInfo: Vulkan hardware and driver information
    """
    result = VulkanInfo(available=False)
    
    vulkaninfo = shutil.which("vulkaninfo")
    if not vulkaninfo:
        # Check Windows registry for Vulkan
        if platform.system() == "Windows" and 'winreg' in globals():
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\Vulkan") as _key:  # type: ignore[attr-defined]
                    result.available = True
                    result.icd_loader_available = True
            except Exception:
                pass
        return result
    
    try:
        output = subprocess.run(
            [vulkaninfo], capture_output=True, text=True, check=False
        )
        if output.returncode == 0:
            result.available = True
            result.icd_loader_available = True
            
            import re
            version_match = re.search(r"Vulkan Instance Version: (\d+\.\d+\.\d+)", output.stdout)
            if version_match:
                result.version = version_match.group(1)
            
            device_matches = re.findall(r"GPU id : (\d+)", output.stdout)
            unique_devices = set(device_matches)
            result.device_count = len(unique_devices)
            
            for section in output.stdout.split("VkPhysicalDeviceProperties:"):
                if not section.strip():
                    continue
                
                name_match = re.search(r"deviceName\s*=\s*(\S.*?)\s*$", section, re.MULTILINE)
                if not name_match:
                    continue
                device_name = name_match.group(1)
                
                device_id_match = re.search(r"deviceID\s*=\s*(\S+)", section)
                device_id = int(device_id_match.group(1), 16) if device_id_match else 0
                
                vendor_match = re.search(r"vendorID\s*=\s*(\S+)", section)
                vendor_id = vendor_match.group(1) if vendor_match else "0x0000"
                
                vendor_map = {
                    "0x1002": "AMD",
                    "0x10DE": "NVIDIA",
                    "0x8086": "Intel",
                    "0x106B": "Apple"
                }
                vendor_name = vendor_map.get(vendor_id.upper(), "Unknown")
                
                memory_match = re.search(
                    r"VkPhysicalDeviceMemoryProperties:\s*memoryHeapCount\s*=\s*(\d+)",
                    section
                )
                total_memory_mb = 0
                
                if memory_match:
                    heap_count = int(memory_match.group(1))
                    for i in range(heap_count):
                        size_match = re.search(
                            rf"memoryHeaps\[{i}\]\.size\s*=\s*(\d+)", 
                            section
                        )
                        if size_match:
                            size_bytes = int(size_match.group(1))
                            size_mb = size_bytes / (1024**2)
                            total_memory_mb = max(total_memory_mb, size_mb)
                
                device_info = {
                    "name": device_name,
                    "device_id": device_id,
                    "vendor": vendor_name,
                    "total_memory_mb": int(total_memory_mb),
                    "free_memory_mb": int(total_memory_mb * 0.8)
                }
                
                result.devices.append(device_info)
    except Exception as e:
        logger.warning(f"Failed to detect Vulkan information: {e}")
    
    return result


def detect_opencl() -> OpenCLInfo:
    """
    Detect OpenCL support.
    
    Returns:
        OpenCLInfo: OpenCL hardware and software information
    """
    result = OpenCLInfo(available=False)
    
    # Attempt to import pyopencl
    try:
        import pyopencl as cl  # type: ignore[import]
        result.available = True
    except ImportError:
        # Fallback to clinfo
        clinfo = shutil.which("clinfo")
        if clinfo:
            try:
                output = subprocess.run(
                    [clinfo],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if output.returncode == 0:
                    result.available = True
                    import re
                    # We no longer store platform_count
                    device_matches = re.findall(r"Device #(\d+):", output.stdout)
                    result.device_count = len(device_matches)
                    
                    version_match = re.search(
                        r"Platform Version\s*:\s*OpenCL (\d+\.\d+)", output.stdout
                    )
                    if version_match:
                        result.version = version_match.group(1)
            except Exception:
                pass
        return result
    
    # If pyopencl is installed, gather info
    try:
        import pyopencl as cl  # type: ignore[import]
        platforms = cl.get_platforms()
        result.device_count = 0
        
        for platform_id, platform_ in enumerate(platforms):
            platform_info = {
                "name": platform_.name,
                "vendor": platform_.vendor,
                "version": platform_.version,
                "profile": platform_.profile,
                "platform_id": platform_id,
                "devices": []
            }
            
            if not result.version:
                import re
                version_match = re.search(r"OpenCL (\d+\.\d+)", platform_.version)
                if version_match:
                    result.version = version_match.group(1)
            
            devices = platform_.get_devices()
            result.device_count += len(devices)
            
            for device_id, device in enumerate(devices):
                total_memory_mb = device.global_mem_size / (1024**2)
                
                device_info = {
                    "name": device.name,
                    "vendor": device.vendor,
                    "device_type": str(device.type).replace("cl.device_type.", ""),
                    "version": device.version,
                    "driver_version": device.driver_version,
                    "max_compute_units": device.max_compute_units,
                    "max_work_group_size": device.max_work_group_size,
                    "total_memory_mb": total_memory_mb,
                    "free_memory_mb": total_memory_mb * 0.8,
                    "platform_id": platform_id,
                    "device_id": device_id
                }
                
                platform_info["devices"].append(device_info)
                result.devices.append({
                    "name": device.name,
                    "vendor": device.vendor,
                    "device_type": str(device.type).replace("cl.device_type.", ""),
                    "total_memory_mb": total_memory_mb,
                    "free_memory_mb": total_memory_mb * 0.8,
                    "platform_name": platform_.name,
                    "platform_id": platform_id,
                    "device_id": device_id
                })
            
            result.platforms.append(platform_info)
    except Exception as e:
        logger.warning(f"Failed to detect OpenCL information: {e}")
    
    return result


def detect_hardware_capabilities() -> HardwareCapabilities:
    """
    Detect all hardware capabilities of the system.
    
    Returns:
        HardwareCapabilities: Complete hardware information
    """
    # Detect CPU capabilities
    cpu_info = detect_cpu_capabilities()
    
    # Detect memory
    memory_info = detect_memory()
    
    # Detect GPU backends
    cuda_info = detect_cuda()
    rocm_info = detect_rocm()
    metal_info = detect_metal()
    vulkan_info = detect_vulkan()
    opencl_info = detect_opencl()
    
    # Compile unified GPU list
    gpus = []
    
    # Add CUDA GPUs
    for device in cuda_info.devices:
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.CUDA,
            name=device["name"],
            vendor="NVIDIA",
            driver_version=cuda_info.driver_version,
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            compute_capability=device["compute_capability"],
            device_id=device["device_id"]
        ))
    
    # Add ROCm GPUs
    for device in rocm_info.devices:
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.ROCM,
            name=device["name"],
            vendor="AMD",
            driver_version=rocm_info.version,
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            compute_capability=device.get("architecture"),
            device_id=device["device_id"]
        ))
    
    # Add Metal GPUs
    for device in metal_info.devices:
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.METAL,
            name=device["name"],
            vendor="Apple",
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            device_id=device["device_id"]
        ))
    
    # Add Vulkan GPUs
    for device in vulkan_info.devices:
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.VULKAN,
            name=device["name"],
            vendor=device["vendor"],
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            device_id=device["device_id"]
        ))
    
    # Add OpenCL GPUs (only if not already added through another backend)
    opencl_device_names = set()
    for device in opencl_info.devices:
        # Skip CPU devices
        if device["device_type"].lower() == "cpu":
            continue
            
        if device["name"] in opencl_device_names:
            continue
        
        opencl_device_names.add(device["name"])
        
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.OPENCL,
            name=device["name"],
            vendor=device["vendor"],
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            device_id=device["device_id"]
        ))
    
    # Determine primary acceleration type
    primary_acceleration = AcceleratorType.CPU
    
    if cuda_info.available and cuda_info.device_count > 0:
        primary_acceleration = AcceleratorType.CUDA
    elif rocm_info.available and rocm_info.device_count > 0:
        primary_acceleration = AcceleratorType.ROCM
    elif metal_info.available:
        primary_acceleration = AcceleratorType.METAL
    elif vulkan_info.available and vulkan_info.device_count > 0:
        primary_acceleration = AcceleratorType.VULKAN
    elif opencl_info.available and opencl_info.device_count > 0:
        primary_acceleration = AcceleratorType.OPENCL
    elif cpu_info.supports_kleidiai:
        primary_acceleration = AcceleratorType.KLEIDIAI
    
    return HardwareCapabilities(
        cpu=cpu_info,
        memory=memory_info,
        primary_acceleration=primary_acceleration,
        gpus=gpus,
        cuda=cuda_info,
        rocm=rocm_info,
        metal=metal_info,
        vulkan=vulkan_info,
        opencl=opencl_info
    )


def get_optimal_llama_cpp_config(hw: HardwareCapabilities) -> LlamaCppConfig:
    """
    Determine optimal configuration for llama.cpp based on hardware.
    
    Args:
        hw: HardwareCapabilities object containing hardware information
        
    Returns:
        LlamaCppConfig: Optimal configuration for llama.cpp
    """
    config = LlamaCppConfig(
        n_threads=max(1, min(hw.cpu.physical_cores - 1, 8)),
        acceleration=hw.primary_acceleration
    )
    
    # Choose acceleration backend
    if hw.primary_acceleration == AcceleratorType.CUDA and hw.cuda.available:
        config.acceleration = AcceleratorType.CUDA
        
        if hw.cuda.device_count > 0:
            main_gpu = 0
            max_memory = 0
            
            for i, device in enumerate(hw.cuda.devices):
                if device["total_memory_mb"] > max_memory:
                    max_memory = device["total_memory_mb"]
                    main_gpu = i
                
            config.main_gpu = main_gpu
            free_memory_gb = hw.cuda.devices[main_gpu]["free_memory_mb"] / 1024
            
            if free_memory_gb > 16:
                config.n_gpu_layers = -1
            elif free_memory_gb > 8:
                config.n_gpu_layers = 35
            elif free_memory_gb > 4:
                config.n_gpu_layers = 20
            else:
                config.n_gpu_layers = 10
            
            if hw.cuda.device_count > 1:
                tensor_split = []
                total_memory = sum(d["free_memory_mb"] for d in hw.cuda.devices)
                for d in hw.cuda.devices:
                    split_ratio = d["free_memory_mb"] / total_memory
                    tensor_split.append(split_ratio)
                config.tensor_split = tensor_split
            
            config.cmake_args = ["-DGGML_CUDA=ON"]
            config.env_vars = {
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(hw.cuda.device_count)),
                "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"
            }
    
    elif hw.primary_acceleration == AcceleratorType.ROCM and hw.rocm.available:
        config.acceleration = AcceleratorType.ROCM
        
        if hw.rocm.device_count > 0:
            config.n_gpu_layers = -1
            config.cmake_args = ["-DGGML_HIP=ON"]
            
            if hw.rocm.architectures:
                config.cmake_args.append(
                    f"-DAMDGPU_TARGETS={';'.join(hw.rocm.architectures)}"
                )
            if hw.rocm.rocm_path:
                config.env_vars["HIP_PATH"] = str(hw.rocm.rocm_path)
            
            config.env_vars["HIP_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(hw.rocm.device_count)
            )
    
    elif hw.primary_acceleration == AcceleratorType.METAL and hw.metal.available:
        config.acceleration = AcceleratorType.METAL
        config.n_gpu_layers = -1
        config.cmake_args = ["-DGGML_METAL=ON"]
    
    elif hw.primary_acceleration == AcceleratorType.VULKAN and hw.vulkan.available:
        config.acceleration = AcceleratorType.VULKAN
        config.n_gpu_layers = -1
        config.cmake_args = ["-DGGML_VULKAN=ON"]
    
    elif hw.primary_acceleration == AcceleratorType.OPENCL and hw.opencl.available:
        config.acceleration = AcceleratorType.OPENCL
        config.n_gpu_layers = -1
        config.cmake_args = ["-DGGML_OPENCL=ON"]
    
    elif hw.primary_acceleration == AcceleratorType.KLEIDIAI and hw.cpu.supports_kleidiai:
        config.acceleration = AcceleratorType.KLEIDIAI
        config.n_gpu_layers = 0
        config.cmake_args = ["-DGGML_CPU_KLEIDIAI=ON"]
        
        if hw.cpu.supports_arm_sve:
            config.env_vars["GGML_KLEIDIAI_SME"] = "1"
    
    else:
        config.acceleration = AcceleratorType.CPU
        config.n_gpu_layers = 0
    
    return config


def find_llama_cpp_executable(paths: DataPaths, acceleration: AcceleratorType) -> Optional[Path]:
    """
    Find existing llama.cpp executable in the specified paths.
    
    Args:
        paths: DataPaths object with directory structure
        acceleration: Type of acceleration to look for
        
    Returns:
        Path to executable if found, None otherwise
    """
    bin_dir = paths.bin
    
    if platform.system() == "Windows":
        extensions = [".exe"]
    else:
        extensions = [""]

    base_names = ["llama-cli", "llama-server", "main"]
    
    # Check for acceleration-specific binaries
    if acceleration != AcceleratorType.CPU:
        for base in base_names:
            for ext in extensions:
                exe_path = bin_dir / f"{base}-{acceleration.value}{ext}"
                if exe_path.exists():
                    return exe_path
    
    # Check for generic binaries
    for base in base_names:
        for ext in extensions:
            exe_path = bin_dir / f"{base}{ext}"
            if exe_path.exists():
                return exe_path
    
    # Check llama.cpp/build directories
    llama_cpp_dir = paths.base / "llama.cpp"
    if llama_cpp_dir.exists():
        build_dir = llama_cpp_dir / "build"
        if build_dir.exists():
            # Check bin subdirectory
            bin_subdir = build_dir / "bin"
            if bin_subdir.exists():
                for base in base_names:
                    for ext in extensions:
                        exe_path = bin_subdir / f"{base}{ext}"
                        if exe_path.exists():
                            return exe_path
            
            # Check Release subdirectory on Windows
            if platform.system() == "Windows":
                release_dir = build_dir / "Release"
                if release_dir.exists():
                    for base in base_names:
                        for ext in extensions:
                            exe_path = release_dir / f"{base}{ext}"
                            if exe_path.exists():
                                return exe_path
            
            # Check root of build dir
            for base in base_names:
                for ext in extensions:
                    exe_path = build_dir / f"{base}{ext}"
                    if exe_path.exists():
                        return exe_path
    
    return None

def get_optimal_model_config(hw: HardwareCapabilities, model_size_gb: float) -> Dict[str, Any]:
    """
    Get optimal configuration for loading and running a model.
    
    Args:
        hw: HardwareCapabilities object with hardware information
        model_size_gb: Size of the model in GB
        
    Returns:
        Dict with optimal configuration parameters
    """
    config = {
        "n_threads": max(1, min(hw.cpu.physical_cores, 8)),
        "n_gpu_layers": 0,
        "use_mlock": True,
        "context_size": 4096,
        "batch_size": 512,
        "use_mmap": True,
        "use_gpu": False
    }
    
    available_memory_gb = hw.memory.available_gb
    
    # If the model is large relative to RAM, try GPU offload if available
    if model_size_gb > available_memory_gb * 0.7:
        if hw.cuda.available and hw.cuda.device_count > 0:
            config["use_gpu"] = True
            largest_gpu = max(hw.cuda.devices, key=lambda d: d["free_memory_mb"])
            free_gpu_gb = largest_gpu["free_memory_mb"] / 1024
            
            if free_gpu_gb > model_size_gb * 0.9:
                config["n_gpu_layers"] = -1
            else:
                ratio = free_gpu_gb / model_size_gb
                config["n_gpu_layers"] = max(1, int(ratio * 64))
        
        elif hw.rocm.available and hw.rocm.device_count > 0:
            config["use_gpu"] = True
            config["n_gpu_layers"] = -1
        
        elif hw.metal.available:
            config["use_gpu"] = True
            config["n_gpu_layers"] = -1
        
        elif hw.vulkan.available and hw.vulkan.device_count > 0:
            config["use_gpu"] = True
            config["n_gpu_layers"] = -1
    
    # Adjust context size for memory
    if available_memory_gb < 8:
        config["context_size"] = 2048
    elif available_memory_gb >= 16:
        config["context_size"] = 8192
    
    # Adjust batch size
    if config["use_gpu"]:
        config["batch_size"] = min(1024, config["context_size"] // 4)
    else:
        config["batch_size"] = min(512, config["context_size"] // 8)
    
    config["use_mmap"] = (model_size_gb > 2)
    config["use_mlock"] = (available_memory_gb > model_size_gb * 1.2)
    
    return config
```

---
### File: src/killeraiagent/models/__init__.py

```python
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
```

---
### File: src/killeraiagent/models/base.py

```python
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
```

---
### File: src/killeraiagent/models/factory.py

```python
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
```

---
### File: src/killeraiagent/models/llama_cpp/__init__.py

```python
from killeraiagent.models.llama_cpp.llama_cpp_cli import LlamaCppCLI
from killeraiagent.models.llama_cpp.llama_cpp_server import LlamaCppServer
from killeraiagent.models.llama_cpp.grammar import GrammarManager
from killeraiagent.models.llama_cpp.templates import Template

__all__ = [
    "LlamaCppCLI",
    "LlamaCppServer",
    "GrammarManager",
    "Template",
]
```

---
### File: src/killeraiagent/models/llama_cpp/grammar.py

```python
"""
Grammar handling for structured generation with llama.cpp.

This module provides utilities for working with GBNF grammars to constrain
model outputs to specific formats like JSON, YAML, etc.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from killeraiagent.paths import get_data_paths


class GrammarManager:
    """Manager for GBNF grammars that constrain model outputs."""
    
    def __init__(self, grammars_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the grammar manager.
        
        Args:
            grammars_dir: Directory containing grammar files (.gbnf)
        """
        self.paths = get_data_paths()
        self.grammars_dir = Path(grammars_dir) if grammars_dir else self.paths.grammars
        
        # Ensure the directory exists
        self.grammars_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache of loaded grammars
        self.grammar_cache = {}
    
    def list_grammars(self) -> List[Dict[str, Any]]:
        """
        List all available grammar files.
        
        Returns:
            List of dictionaries with grammar information
        """
        grammars = []
        
        # Search for .gbnf files
        grammar_files = list(self.grammars_dir.glob("**/*.gbnf"))
        
        for file_path in grammar_files:
            info = {
                "name": file_path.stem,
                "path": str(file_path),
                "relative_path": str(file_path.relative_to(self.grammars_dir)),
                "size": file_path.stat().st_size
            }
            
            # Try to determine purpose from first few lines
            try:
                with open(file_path, 'r') as f:
                    first_lines = [next(f) for _ in range(5) if f]
                    for line in first_lines:
                        if line.strip().startswith('#'):
                            info["description"] = line.strip('# \n')
                            break
            except Exception:
                pass
            
            grammars.append(info)
        
        return grammars
    
    def get_grammar_path(self, grammar_name: str) -> Optional[Path]:
        """
        Get the path to a grammar file by name.
        
        Args:
            grammar_name: Name of the grammar (with or without .gbnf extension)
        
        Returns:
            Path to the grammar file or None if not found
        """
        if not grammar_name.endswith('.gbnf'):
            grammar_name += '.gbnf'
        
        exact_path = self.grammars_dir / grammar_name
        if exact_path.exists():
            return exact_path
        
        for file_path in self.grammars_dir.glob("**/*.gbnf"):
            if file_path.name == grammar_name:
                return file_path
        
        return None
    
    def get_or_create_json_grammar(self, schema: Dict[str, Any]) -> Path:
        """
        Generate a GBNF grammar from a JSON schema and save it.
        
        Args:
            schema: JSON schema definition
        
        Returns:
            Path to the generated grammar file
        """
        import hashlib
        schema_str = json.dumps(schema, sort_keys=True)
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:10]
        grammar_name = f"json_schema_{schema_hash}.gbnf"
        
        grammar_path = self.grammars_dir / grammar_name
        if grammar_path.exists():
            return grammar_path
        
        grammar_content = self._generate_json_grammar(schema)
        
        with open(grammar_path, 'w') as f:
            f.write(grammar_content)
        
        return grammar_path
    
    def _generate_json_grammar(self, schema: Dict[str, Any]) -> str:
        """
        Generate GBNF grammar from JSON schema.
        
        Args:
            schema: JSON schema definition
        
        Returns:
            GBNF grammar text
        """
        lines = [
            "# JSON grammar generated from schema",
            "root ::= object",
            "",
            "object ::= \"{\" ws (pair (ws \",\" ws pair)*)? ws \"}\"",
            "pair ::= string ws \":\" ws value",
            "",
            "array ::= \"[\" ws (value (ws \",\" ws value)*)? ws \"]\"",
            "",
            "value ::= object | array | string | number | boolean | null",
            "",
            "string ::= \"\\\"\" ([^\\\"\\\\] | \\\\\\\\ | \\\\\\\")* \"\\\"\"",
            "",
            "number ::= integer | float",
            "integer ::= [\"-\"]? (\"0\" | [1-9] [0-9]*)",
            "float ::= [\"-\"]? (\"0\" | [1-9] [0-9]*) \".\" [0-9]+",
            "",
            "boolean ::= \"true\" | \"false\"",
            "null ::= \"null\"",
            "",
            "# Whitespace handling",
            "ws ::= [ \\t\\n]*"
        ]
        
        if 'properties' in schema:
            property_rules = []            
            for prop_name, prop_schema in schema['properties'].items():
                if 'type' in prop_schema:
                    prop_type = prop_schema['type']
                    if prop_type == 'string':
                        if 'enum' in prop_schema:
                            enum_values = " | ".join([f'\"\\\"{val}\\\"\"' for val in prop_schema['enum']])
                            property_rules.append(f"{prop_name}_value ::= {enum_values}")
                        else:
                            property_rules.append(f"{prop_name}_value ::= string")
                    elif prop_type in ['number', 'integer']:
                        property_rules.append(f"{prop_name}_value ::= number")
                    elif prop_type == 'boolean':
                        property_rules.append(f"{prop_name}_value ::= boolean")
                    elif prop_type == 'array':
                        property_rules.append(f"{prop_name}_value ::= array")
                    elif prop_type == 'object':
                        property_rules.append(f"{prop_name}_value ::= object")
                    else:
                        property_rules.append(f"{prop_name}_value ::= value")
            
            if property_rules:
                lines.append("")
                lines.append("# Property-specific rules")
                lines.extend(property_rules)
        
        return "\n".join(lines)


def create_grammar_manager() -> GrammarManager:
    """Factory function to create a grammar manager."""
    return GrammarManager()
```

---
### File: src/killeraiagent/models/llama_cpp/llama_cpp_cli.py

```python
"""
LlamaCpp CLI implementation for KAIA.

This module provides a Python interface to the llama.cpp CLI executable.
"""

import os
import sys
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Tuple, Optional, Iterator, Callable

from killeraiagent.models.base import LLM, ModelInfo
from killeraiagent.paths import get_data_paths
from killeraiagent.hardware import AcceleratorType, detect_hardware_capabilities

logger = logging.getLogger(__name__)

class LlamaCppCLI(LLM):
    """
    Interface to llama.cpp CLI executable for inference.
    
    This class manages the subprocess for the llama-cli executable.
    """
    
    def __init__(
        self,
        model_info: ModelInfo,
        n_gpu_layers: int = 0,
        n_threads: int = 4,
        context_size: int = 4096,
        **kwargs
    ):
        """
        Initialize LlamaCpp CLI interface.
        
        Args:
            model_info: Model information
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            n_threads: Number of CPU threads to use
            context_size: Context window size in tokens
            **kwargs: Additional arguments
        """
        super().__init__(model_info, **kwargs)
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.context_size = context_size
        self.executable_path = None
        self.process = None
    
    def _find_executable(self) -> Optional[Path]:
        """
        Find the llama.cpp executable path.
        
        Returns:
            Path to executable or None if not found
        """
        paths = get_data_paths()
        bin_dir = paths.bin
        
        # Extensions based on platform
        if sys.platform == "win32":
            extensions = [".exe"]
        else:
            extensions = [""]
        
        # Try to find acceleration-specific executables first
        hw = detect_hardware_capabilities()
        acc_type = hw.primary_acceleration
        
        # Search for executables with the pattern: llama-cli[-ACCEL_TYPE][.exe]
        for ext in extensions:
            # Check for acceleration-specific binary
            if acc_type != AcceleratorType.CPU:
                exe_path = bin_dir / f"llama-cli-{acc_type.value}{ext}"
                if exe_path.exists():
                    logger.info(f"Found acceleration-specific executable: {exe_path}")
                    return exe_path
            
            # Check for generic binary
            exe_path = bin_dir / f"llama-cli{ext}"
            if exe_path.exists():
                logger.info(f"Found generic executable: {exe_path}")
                return exe_path
            
            # Check for alternative names
            for alt_name in ["main", "llama-main", "llama"]:
                exe_path = bin_dir / f"{alt_name}{ext}"
                if exe_path.exists():
                    logger.info(f"Found alternative executable: {exe_path}")
                    return exe_path
        
        logger.warning("No llama.cpp executable found")
        return None
    
    def load(self) -> bool:
        """
        Load the model. For CLI interface, this just checks if the executable exists.
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_loaded:
            return True
        
        # Check if model path exists
        if not self.model_info.model_path:
            logger.error("Model path not specified")
            return False
        
        model_path = Path(self.model_info.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Find executable
        self.executable_path = self._find_executable()
        if not self.executable_path:
            logger.error("llama.cpp executable not found. Please build or install it.")
            return False
        
        self.is_loaded = True
        return True
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> Tuple[str, Any]:
        """
        Generate text using llama.cpp CLI.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeated tokens
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        if not self.is_loaded:
            if not self.load():
                raise RuntimeError("Failed to load model")
        
        # Save prompt to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            prompt_file = f.name
            f.write(prompt)
        
        try:
            # Build command
            cmd = [
                str(self.executable_path),
                "-m", str(self.model_info.model_path),
                "-f", prompt_file,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "--top_p", str(top_p),
                "--repeat_penalty", str(repeat_penalty),
                "-t", str(self.n_threads),
                "-c", str(self.context_size),
                "--color", "0",  # Disable color output
                "-ngl", str(self.n_gpu_layers)
            ]
            
            # Add any additional arguments from kwargs
            for key, value in kwargs.items():
                if key == "top_k":
                    cmd.extend(["--top_k", str(value)])
                elif key == "frequency_penalty":
                    cmd.extend(["--frequency_penalty", str(value)])
                elif key == "presence_penalty":
                    cmd.extend(["--presence_penalty", str(value)])
                elif key == "mirostat":
                    cmd.extend(["--mirostat", str(value)])
                elif key == "mirostat_tau":
                    cmd.extend(["--mirostat_tau", str(value)])
                elif key == "mirostat_eta":
                    cmd.extend(["--mirostat_eta", str(value)])
                elif key == "seed":
                    cmd.extend(["--seed", str(value)])
            
            # Run process
            logger.info(f"Running llama.cpp: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            
            # Check for errors
            if result.returncode != 0:
                error_msg = f"llama.cpp process failed with code {result.returncode}: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Extract output (removing the prompt part)
            full_output = result.stdout.strip()
            
            # Skip the prompt part (everything before the first newline)
            output_lines = full_output.split("\n")
            if len(output_lines) > 1:
                # Skip lines until we're past the prompt
                prompt_completed = False
                result_lines = []
                
                for line in output_lines:
                    if not prompt_completed and line.strip() == "":
                        prompt_completed = True
                        continue
                    
                    if prompt_completed:
                        result_lines.append(line)
                
                # If we never found the end of the prompt, use the last 75% of the output
                if not prompt_completed:
                    output = full_output[int(len(full_output) * 0.25):]
                else:
                    output = "\n".join(result_lines)
            else:
                # If there's only one line, assume it's all output
                output = full_output
            
            # Metadata
            metadata = {
                "tokens": len(output.split()),
                "model": self.model_info.model_id,
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                    **kwargs
                }
            }
            
            return output, metadata
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(prompt_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary prompt file: {e}")
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        on_token: Optional[Callable] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream generated text using llama.cpp CLI.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeated tokens
            on_token: Callback function for each token
            **kwargs: Additional arguments
            
        Yields:
            Generated tokens
        """
        if not self.is_loaded:
            if not self.load():
                raise RuntimeError("Failed to load model")
        
        # Save prompt to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            prompt_file = f.name
            f.write(prompt)
        
        try:
            # Build command
            cmd = [
                str(self.executable_path),
                "-m", str(self.model_info.model_path),
                "-f", prompt_file,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "--top_p", str(top_p),
                "--repeat_penalty", str(repeat_penalty),
                "-t", str(self.n_threads),
                "-c", str(self.context_size),
                "--color", "0",  # Disable color output
                "-ngl", str(self.n_gpu_layers),
                "--streaming", "1"  # Enable streaming
            ]
            
            # Add any additional arguments from kwargs
            for key, value in kwargs.items():
                if key == "top_k":
                    cmd.extend(["--top_k", str(value)])
                elif key == "frequency_penalty":
                    cmd.extend(["--frequency_penalty", str(value)])
                elif key == "presence_penalty":
                    cmd.extend(["--presence_penalty", str(value)])
                elif key == "mirostat":
                    cmd.extend(["--mirostat", str(value)])
                elif key == "mirostat_tau":
                    cmd.extend(["--mirostat_tau", str(value)])
                elif key == "mirostat_eta":
                    cmd.extend(["--mirostat_eta", str(value)])
                elif key == "seed":
                    cmd.extend(["--seed", str(value)])
            
            # Run process
            logger.info(f"Running llama.cpp stream: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            
            # Store process for cleanup
            self.process = process
            
            # Skip the prompt part 
            prompt_completed = False
            prompt_len = len(prompt)
            chunk_buffer = ""
            
            # Read from stdout in real-time
            try:
                if process.stdout is None:
                    raise RuntimeError("Failed to start process or capture output stream")
                
                for line in process.stdout:
                    if not prompt_completed:
                        chunk_buffer += line
                        # Check if we've buffered more than the prompt length
                        if len(chunk_buffer) >= prompt_len:
                            # Skip past the prompt
                            excess = chunk_buffer[prompt_len:]
                            if excess:
                                if on_token:
                                    on_token(excess)
                                yield excess
                            prompt_completed = True
                            chunk_buffer = ""
                    else:
                        if on_token:
                            on_token(line)
                        yield line
            
            finally:
                # Make sure process is terminated
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                
                # Check for errors
                stderr = process.stderr.read() if process.stderr else ""
                if stderr and process.returncode != 0:
                    logger.error(f"llama.cpp process error: {stderr}")
                
                self.process = None
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(prompt_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary prompt file: {e}")
    
    def unload(self) -> None:
        """Unload the model and clean up resources."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        self.process = None
        self.is_loaded = False
```

---
### File: src/killeraiagent/models/llama_cpp/llama_cpp_server.py

```python
"""
Server-based implementation of llama.cpp model (LlamaCppServer).

Keeps the model loaded in memory as a server, for faster repeated requests.
"""

import os
import time
import logging
import subprocess
import requests
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from killeraiagent.models.base import LLM, ModelInfo
from killeraiagent.setup.setup_core import get_data_paths
from killeraiagent.hardware import detect_hardware_capabilities

logger = logging.getLogger(__name__)


class LlamaCppServer(LLM):
    """
    Implementation of llama.cpp as a persistent server process
    with an OpenAI-compatible API.
    """

    def __init__(
        self,
        model_info: ModelInfo,
        chat_format: str = "chatml",
        chat_template: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        seed: int = -1,
        grammar_file: Optional[str] = None,
        server_host: str = "127.0.0.1",
        server_port: int = 8080,
        n_parallel: int = 1,
        auto_start: bool = False
    ):
        """
        Args:
            model_info: Metadata about the model
            chat_format: Chat format style (e.g. 'chatml', 'llama2', etc.)
            chat_template: Jinja2 template for chat
            n_ctx: Context length
            n_threads: CPU threads
            n_gpu_layers: GPU-accelerated layers
            seed: Random seed
            grammar_file: Optional grammar file path
            server_host: Host for the local server
            server_port: Port for the local server
            n_parallel: How many requests to handle in parallel
            auto_start: If True, start the server in constructor
        """
        super().__init__(model_info)
        self.chat_format = chat_format
        self.chat_template = chat_template

        self.n_ctx = n_ctx if n_ctx is not None else model_info.context_length
        self.n_threads = n_threads if n_threads is not None else (os.cpu_count() or 4)
        self.n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None
            else (-1 if model_info.requires_gpu else 0)
        )
        self.seed = seed
        self.grammar_file = grammar_file

        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        self.n_parallel = n_parallel

        self.paths = get_data_paths()
        self.bin_dir = self.paths.bin
        self.temp_dir = self.paths.temp

        # Path to the llama server executable
        self.llama_server_path = self.bin_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if not self.llama_server_path.exists():
            self.llama_server_path = self.bin_dir / ("server.exe" if os.name == "nt" else "server")

        # Persistent process
        self.server_process: Optional[subprocess.Popen] = None
        self.health_check_thread: Optional[threading.Thread] = None
        self.health_check_stop = threading.Event()

        # Adjust GPU layers
        self._adjust_gpu_layers()

        if auto_start:
            ok = self.load()
            if not ok:
                logger.error("Failed to auto-start the llama.cpp server.")

    def _adjust_gpu_layers(self) -> None:
        hw_caps = detect_hardware_capabilities()
        acc_type = hw_caps.primary_acceleration.value
        if acc_type == "cpu" and self.n_gpu_layers != 0:
            self.n_gpu_layers = 0
            logger.info("CPU-only acceleration, set n_gpu_layers=0")
        if acc_type == "metal" and self.n_gpu_layers > 0:
            logger.info(f"Metal acceleration with {self.n_gpu_layers} GPU layers")

    def load(self) -> bool:
        """
        Start the server if not already running.
        Returns True if successfully started/connected, else False.
        """
        if self.is_loaded and self.server_process and self.server_process.poll() is None:
            return True

        # Check model file and server exe
        if not self.model_info.model_path or not Path(self.model_info.model_path).exists():
            logger.error(f"Model file not found: {self.model_info.model_path}")
            return False
        if not self.llama_server_path.exists():
            logger.error(f"Server executable not found: {self.llama_server_path}")
            return False

        return self._start_server()

    def _start_server(self) -> bool:
        if self.server_process is not None:
            logger.warning("Server already running, stopping first.")
            self.close()

        cmd = self._build_server_command()
        logger.info(f"Starting llama.cpp server: {' '.join(cmd)}")

        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            logger.error(f"Failed to start server process: {e}")
            self.server_process = None
            return False

        # Wait up to 30s for /v1/models to return 200
        start_time = time.time()
        while time.time() - start_time < 30:
            if self.server_process.poll() is not None:
                err_out = ""
                if self.server_process.stderr:
                    err_out = self.server_process.stderr.read()
                logger.error(f"Server exited early with code {self.server_process.returncode}\n{err_out}")
                return False

            try:
                resp = requests.get(f"{self.server_url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    logger.info("Server started successfully.")
                    self.is_loaded = True
                    # Start health check thread
                    self._start_health_check_thread()
                    return True
            except requests.RequestException:
                pass

            time.sleep(0.5)

        logger.error("Timed out waiting for server to respond.")
        return False

    def _start_health_check_thread(self) -> None:
        if self.health_check_thread and self.health_check_thread.is_alive():
            return
        self.health_check_stop.clear()
        self.health_check_thread = threading.Thread(
            target=self._health_check_worker,
            daemon=True
        )
        self.health_check_thread.start()

    def _health_check_worker(self, interval: int = 30) -> None:
        while not self.health_check_stop.is_set():
            if self.server_process and self.server_process.poll() is not None:
                code = self.server_process.returncode
                logger.warning(f"Server died (code {code}); restarting...")
                self._start_server()
            else:
                try:
                    resp = requests.get(f"{self.server_url}/v1/models", timeout=2)
                    if resp.status_code != 200:
                        logger.warning(f"Health check got status {resp.status_code}, restarting.")
                        self._start_server()
                except requests.RequestException:
                    logger.warning("Health check request failed, restarting server.")
                    self._start_server()
            self.health_check_stop.wait(interval)

    def close(self) -> None:
        """
        Stop the server and free resources.
        """
        if self.health_check_thread:
            self.health_check_stop.set()
            self.health_check_thread.join(timeout=2)
        if self.server_process is not None:
            logger.info("Stopping llama.cpp server.")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, forcing kill.")
                self.server_process.kill()
            self.server_process = None
        self.is_loaded = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1
    ) -> Tuple[str, Any]:
        """
        Call the local server /v1/completions (for a standard prompt).
        """
        if not self.load():
            return ("Error: LlamaCppServer not running", None)

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty - 1.0,
            "stream": False
        }

        try:
            resp = requests.post(f"{self.server_url}/v1/completions", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices"):
                return (data["choices"][0].get("text", ""), data)
            return ("", data)
        except Exception as e:
            logger.error(f"Error calling server: {e}")
            return (f"Error: {e}", {"error": str(e)})

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1
    ) -> Dict[str, Any]:
        """
        Call /v1/chat/completions with a list of role/content messages.
        """
        if not self.load():
            return {"error": "Server not running, cannot generate chat completion"}

        payload = {
            "model": self.model_info.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty - 1.0,
            "stream": False
        }

        try:
            resp = requests.post(f"{self.server_url}/v1/chat/completions", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error in generate_chat_completion: {e}")
            return {"error": str(e)}

    def _build_server_command(self) -> List[str]:
        cmd = [
            str(self.llama_server_path),
            "-m", str(self.model_info.model_path),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-ngl", str(self.n_gpu_layers),
            "--host", self.server_host,
            "--port", str(self.server_port),
        ]

        if self.n_parallel > 1:
            cmd.extend(["-np", str(self.n_parallel)])
        if self.chat_format:
            cmd.extend(["--chat-format", self.chat_format])
        if self.chat_template:
            template_file = self.temp_dir / f"server_template_{int(time.time())}.txt"
            with open(template_file, 'w') as f:
                f.write(self.chat_template)
            cmd.extend(["--chat-template", str(template_file)])
        if self.grammar_file:
            cmd.extend(["--grammar-file", str(self.grammar_file)])

        return cmd
    
    def unload(self) -> None:
        """Implementation of abstract unload method. Nothing to unload for API-based models."""
        self.is_loaded = False
```

---
### File: src/killeraiagent/models/llama_cpp/templates.py

```python
"""
Chat template handling for llama.cpp models.

This module provides utilities for working with Jinja2-based chat templates
that format messages for different model types.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from jinja2 import Template, Environment, BaseLoader, TemplateError, meta

from killeraiagent.paths import get_data_paths


class ChatTemplateManager:
    """Manager for chat templates used with different models."""
    
    BUILT_IN_TEMPLATES = {
        "chatml": """{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>'}}
{% endfor %}
<|im_start|>assistant
""",
        
        "llama2": """{% if messages[0]['role'] == 'system' %}
<s>[INST] <<SYS>>
{{ messages[0]['content'] }}
<</SYS>>

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif %}

{% for message in loop_messages %}
{% if message['role'] == 'user' %}
{% if loop.first %}
<s>[INST] {{ message['content'] }} [/INST]
{% else %}
[INST] {{ message['content'] }} [/INST]
{% endif %}
{% elif message['role'] == 'assistant' %}
 {{ message['content'] }} </s>
{% endif %}
{% endfor %}
""",
        
        "mistral": """{% for message in messages %}
{% if message['role'] == 'user' %}
[INST] {{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}
{{ message['content'] }}
{% elif message['role'] == 'system' %}
<s>[INST] <<SYS>>
{{ message['content'] }}
<</SYS>>
{% endif %}
{% endfor %}
""",
        
        "alpaca": """{% if messages[0]['role'] == 'system' %}
{{ messages[0]['content'] }}

{% endif %}
{% for message in messages %}
{% if message['role'] == 'user' %}
### Instruction:
{{ message['content'] }}

{% elif message['role'] == 'assistant' and not loop.last %}
### Response:
{{ message['content'] }}

{% endif %}
{% endfor %}
### Response:
"""
    }
    
    def __init__(self, templates_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory containing custom template files
        """
        self.paths = get_data_paths()
        self.templates_dir = Path(templates_dir) if templates_dir else self.paths.templates
        
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.template_cache = {}
        
        self._create_builtin_templates()
    
    def _create_builtin_templates(self):
        """Create files for built-in templates if they don't exist."""
        for name, content in self.BUILT_IN_TEMPLATES.items():
            template_path = self.templates_dir / f"{name}.jinja2"
            if not template_path.exists():
                with open(template_path, 'w') as f:
                    f.write(content)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of dictionaries with template information
        """
        templates = []
        
        for name in self.BUILT_IN_TEMPLATES:
            templates.append({
                "name": name,
                "type": "built-in",
                "path": str(self.templates_dir / f"{name}.jinja2")
            })
        
        for file_path in self.templates_dir.glob("*.jinja2"):
            name = file_path.stem
            if name not in self.BUILT_IN_TEMPLATES:
                templates.append({
                    "name": name,
                    "type": "custom",
                    "path": str(file_path)
                })
        
        return templates
    
    def get_template(self, template_name: str) -> Template:
        """
        Get a Jinja2 template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Jinja2 Template object
            
        Raises:
            ValueError: If template not found
        """
        if template_name in self.template_cache:
            return self.template_cache[template_name]
        
        if template_name in self.BUILT_IN_TEMPLATES:
            template_str = self.BUILT_IN_TEMPLATES[template_name]
            template = Template(template_str)
            self.template_cache[template_name] = template
            return template
        
        template_path = self.templates_dir / f"{template_name}.jinja2"
        if not template_path.exists():
            raise ValueError(f"Template not found: {template_name}")
        
        with open(template_path, 'r') as f:
            template_str = f.read()
        
        template = Template(template_str)
        self.template_cache[template_name] = template
        return template
    
    def render_template(self, template_name: str, messages: List[Dict[str, str]]) -> str:
        template = self.get_template(template_name)
        return template.render(messages=messages)
    
    def detect_format(self, model_path: Union[str, Path]) -> str:
        model_name = str(model_path).lower()
        
        if "llama-2" in model_name or "llama2" in model_name:
            return "llama2"
        elif "mistral" in model_name or "mixtral" in model_name:
            return "mistral"
        elif "alpaca" in model_name or "vicuna" in model_name:
            return "alpaca"
        
        return "chatml"
    
    def create_template(self, name: str, content: str) -> str:
        try:
            Template(content)
        except TemplateError as e:
            raise ValueError(f"Invalid template: {e}")
        
        if name in self.BUILT_IN_TEMPLATES:
            raise ValueError(f"Cannot override built-in template: {name}")
        
        template_path = self.templates_dir / f"{name}.jinja2"
        with open(template_path, 'w') as f:
            f.write(content)
        
        if name in self.template_cache:
            del self.template_cache[name]
        
        return str(template_path)
    
    def get_template_content(self, template_name: str) -> str:
        if template_name in self.BUILT_IN_TEMPLATES:
            return self.BUILT_IN_TEMPLATES[template_name]
        
        template_path = self.templates_dir / f"{template_name}.jinja2"
        if not template_path.exists():
            raise ValueError(f"Template not found: {template_name}")
        
        with open(template_path, 'r') as f:
            return f.read()
    
    def analyze_template(self, template_name: str) -> Dict[str, Any]:
        content = self.get_template_content(template_name)
        
        env = Environment(loader=BaseLoader())
        ast = env.parse(content)
        variables = meta.find_undeclared_variables(ast)
        
        return {
            "name": template_name,
            "variables": list(variables),
            "length": len(content),
            "has_messages_loop": ("messages" in variables and "{% for" in content and "messages" in content)
        }


def create_template_manager() -> ChatTemplateManager:
    return ChatTemplateManager()


def format_messages(
    messages: List[Dict[str, str]], 
    template_name_or_content: str,
    is_template_content: bool = False
) -> str:
    if is_template_content:
        template = Template(template_name_or_content)
        return template.render(messages=messages)
    else:
        manager = create_template_manager()
        return manager.render_template(template_name_or_content, messages)
```

---
### File: src/killeraiagent/models/openai/__init__.py

```python
"""
OpenAI LLM module initialization.
"""

from .openai_llm import OpenAILLM

__all__ = ["OpenAILLM"]
```

---
### File: src/killeraiagent/models/openai/openai_llm.py

```python
"""
OpenAI-based LLM class (OpenAILLM).

Uses the remote OpenAI API for completions and chat, no local resources to manage.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import requests

from killeraiagent.models.base import LLM, ModelInfo

logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    """
    LLM for the OpenAI remote API (chat/completions).
    """

    def __init__(
        self,
        model_info: ModelInfo,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        default_model: Optional[str] = None,
        request_timeout: int = 120
    ):
        """
        Args:
            model_info: Basic info about the model (though we mostly use default_model).
            api_key: The OpenAI API key (Bearer token).
            base_url: The base URL for the API (can override for dev or proxies).
            default_model: e.g. "gpt-3.5-turbo"
            request_timeout: Request timeout in seconds.
        """
        super().__init__(model_info)
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model or model_info.model_id
        self.request_timeout = request_timeout

    def load(self) -> bool:
        """
        No local resources. We just mark it loaded if we have an API key.
        """
        if self.is_loaded:
            return True
        if not self.api_key:
            logger.warning("OpenAI API key not provided.")
        self.is_loaded = True
        return True

    def close(self) -> None:
        """
        No local resources to free. Just reset is_loaded if you want.
        """
        self.is_loaded = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0
    ) -> Tuple[str, Any]:
        """
        Create a simple text completion via /v1/completions.
        """
        if not self.load():
            return ("Error: No API key or not loaded", None)

        url = f"{self.base_url}/completions"
        headers = self._get_headers()
        payload = {
            "model": self.default_model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            # repeat_penalty is not exactly in OpenAI API,
            # you might approximate via frequency_penalty or presence_penalty
            "frequency_penalty": repeat_penalty - 1.0 if repeat_penalty != 1.0 else 0.0
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices"):
                return (data["choices"][0].get("text", ""), data)
            return ("", data)
        except Exception as e:
            logger.error(f"OpenAI generate error: {e}")
            return (f"Error: {e}", {"error": str(e)})

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create a chat completion via /v1/chat/completions.
        """
        if not self.load():
            return {"error": "OpenAILLM not loaded or no API key"}

        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        payload = {
            "model": self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty - 1.0 if repeat_penalty != 1.0 else 0.0
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"OpenAI chat completion error: {e}")
            return {"error": str(e)}

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def unload(self) -> None:
        """Implementation of abstract unload method. Nothing to unload for API-based models."""
        self.is_loaded = False
```

---
### File: src/killeraiagent/models/transformers/__init__.py

```python
"""
Initialize the Transformers-based models module.
"""

from killeraiagent.models.transformers.huggingface_llm import TransformersLLM

__all__ = [
    "TransformersLLM"
]
```

---
### File: src/killeraiagent/models/transformers/huggingface_llm.py

```python
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
```

---
### File: src/killeraiagent/multiturn_teapot.py

```python
"""
multiturn_teapot.py

Transforms a Teapot-based model (Flan-T5 variant) into a multi-turn chat agent with retrieval augmentation (RAG).
It loads a local FAQ file (.jsonl), builds embeddings using a feature-extraction pipeline, and retrieves the top-k FAQ snippets
to augment the conversation prompt.

The embedding model is now set to "teapotai/teapotembedding" with truncation enabled.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from killeraiagent.models.base import ModelInfo
from killeraiagent.models.transformers.huggingface_llm import TransformersLLM
from killeraiagent.paths import get_data_paths
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Teapot, an AI assistant specialized in KAIA setup and configuration. "
    "Answer questions succinctly and provide practical guidance. If you find relevant FAQ snippets, "
    "incorporate them into your response."
)

# Documentation lookup dictionary.
# Adjust the paths as needed to point to the actual documentation man pages.
BASE_DOCUMENTATION_PATH = get_data_paths().base
DOCUMENTATION_PATHS = {
    "llama": BASE_DOCUMENTATION_PATH / "docs" / "llamacpp_usage.md",
    "setup": BASE_DOCUMENTATION_PATH / "docs" / "kaia_setup.md",
    "teapot": BASE_DOCUMENTATION_PATH / "docs" / "teapot_usage.md"
}

class FAQEntry:
    """Structure for a single FAQ record."""
    def __init__(self, question: str, answer: str) -> None:
        self.question = question
        self.answer = answer
        self.embedding: Optional[np.ndarray] = None

class MultiturnTeapot:
    """
    Multi-turn chat agent for KAIA that uses a Transformers-based Teapot model and RAG over a local FAQ file.
    """
    def __init__(
        self,
        faq_path: Optional[Path] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_context_tokens: int = 1024,
        rag_top_k: int = 2,
        model_info: Optional[ModelInfo] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.rag_top_k = rag_top_k
        self.conversation_history: List[Dict[str, str]] = []
        self.rag_context: str = ""

        if model_info is None:
            from killeraiagent.models.base import ModelInfo
            model_info = ModelInfo(
                model_id="teapotllm",
                model_engine="transformers",
                description="TeapotLLM with RAG for KAIA",
                context_length=1024,
                requires_gpu=False,
                model_size_gb=1.0
            )
        self.llm = TransformersLLM(
            model_info=model_info,
            pipeline_type="text2text-generation",
            model_name_or_path="teapotai/teapotllm",
            chat_format="flan"
        )
        self.llm.load()

        if faq_path is None:
            from killeraiagent.setup.setup_core import get_data_paths
            paths = get_data_paths()
            faq_path = paths.base / "faq" / "kaia_faq.jsonl"
        self.faq_path = faq_path
        self.faq_entries: List[FAQEntry] = []
        self._embedding_pipeline = None

        self._load_faq()
        self._build_embeddings()

    def _load_faq(self) -> None:
        """Loads FAQ entries from a JSON Lines file."""
        if not self.faq_path.exists():
            logger.warning(f"FAQ file not found at {self.faq_path}")
            return
        with open(self.faq_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                question = data.get("question", "")
                answer = data.get("answer", "")
                if question and answer:
                    self.faq_entries.append(FAQEntry(question, answer))
        logger.info(f"Loaded {len(self.faq_entries)} FAQ entries from {self.faq_path}")

    def _build_embeddings(self) -> None:
        """Build embeddings for each FAQ entry using a feature-extraction pipeline."""
        if not self.faq_entries:
            return
        logger.info("Building embeddings for FAQ entries using teapotai/teapotembedding.")
        try:
            self._embedding_pipeline = pipeline(
                "feature-extraction",
                model="teapotai/teapotembedding",
                truncation=True,
                device=-1  # CPU mode; adjust if necessary.
            )
            for entry in self.faq_entries:
                text_to_embed = entry.question + " " + entry.answer
                result = self._embedding_pipeline(text_to_embed)
                if result is not None:
                    emb_output = result if isinstance(result, list) else [result]
                    emb_array = np.array(emb_output[0])
                    vec = np.mean(emb_array, axis=0)
                    entry.embedding = vec
        except Exception as e:
            logger.error(f"Error building FAQ embeddings: {e}")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def _handle_doc_request(self, query: str) -> str:
        """
        Handle documentation requests if the query starts with "doc" or "man".
        Looks up the topic and returns the corresponding documentation file path.
        """
        parts = query.strip().split()
        if len(parts) >= 2:
            topic = parts[1].lower()
            doc_path = DOCUMENTATION_PATHS.get(topic)
            if doc_path:
                return f"Documentation for '{topic}' is available at: {doc_path}"
            else:
                return f"Sorry, no documentation found for topic '{topic}'. Available topics are: {list(DOCUMENTATION_PATHS.keys())}"
        else:
            return "Usage: 'doc <topic>' (e.g. 'doc llama')."

    def chat(self, user_input: str) -> str:
        """Process a user query, retrieve relevant FAQ snippets, and generate a response.
           Also handles documentation lookup if requested."""
        # Check if user asked for documentation help:
        if user_input.strip().lower().startswith(("doc ", "man ")):
            return self._handle_doc_request(user_input)

        self.add_message("user", user_input)
        relevant_faqs = self._retrieve_faq(user_input, self.rag_top_k)
        prompt = self._format_prompt(user_input, relevant_faqs)
        text, _ = self.llm.generate(
            prompt,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.0,
            do_sample=True
        )
        text = text.strip()
        self.add_message("assistant", text)
        return text

    def _retrieve_faq(self, query: str, k: int) -> List[str]:
        """Retrieve top-k FAQ snippets based on cosine similarity with the query."""
        if not self.faq_entries or not self._embedding_pipeline:
            return []
        result = self._embedding_pipeline(query)
        if isinstance(result, list):
            query_emb = result[0]
        else:
            query_emb = np.array(result)
        query_vec = np.mean(query_emb, axis=0)
        sims = []
        for entry in self.faq_entries:
            if entry.embedding is None:
                continue
            sim = cosine_similarity([query_vec], [entry.embedding])[0][0]
            sims.append((sim, entry))
        sims.sort(key=lambda x: x[0], reverse=True)
        top_entries = [entry for sim, entry in sims if sim >= 0.2][:k]
        snippets = [f"(Q) {entry.question}\n(A) {entry.answer}" for entry in top_entries]
        return snippets

    def _format_prompt(self, user_input: str, faq_snippets: List[str]) -> str:
        """Construct the prompt using system prompt, additional RAG context, FAQ snippets, and conversation history."""
        history_lines = []
        for msg in self.conversation_history[:-1]:
            history_lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
        history_lines.append(f"User: {user_input}\nAssistant:")
        snippet_block = ""
        if faq_snippets:
            snippet_block = "[Relevant FAQ Snippets]\n" + "\n\n".join(faq_snippets) + "\n\n"
        prompt = f"{self.system_prompt}\n\n{self.rag_context}\n\n{snippet_block}" + "\n".join(history_lines)
        return prompt

    def close(self) -> None:
        """Close the underlying LLM."""
        self.llm.close()
```

---
### File: src/killeraiagent/paths.py

```python
from pathlib import Path
from pydantic import BaseModel, Field, validator

class DataPaths(BaseModel):
    """Paths for application data storage."""
    base: Path
    models: Path = Field()
    logs: Path = Field()
    config: Path = Field()
    temp: Path = Field()
    grammars: Path = Field()
    templates: Path = Field()
    docs: Path = Field()
    bin: Path = Field()
    
    @validator("models", "logs", "config", "bin", pre=True, always=True)
    def set_derived_paths(cls, v, values):
        """Set derived paths if not explicitly provided."""
        if v is None and "base" in values:
            return values["base"] / "models"
        return v
    
    def ensure_dirs_exist(self):
        """Create all directories if they don't exist."""
        for path in [self.base, self.models, self.logs, self.config, self.bin]:
            path.mkdir(parents=True, exist_ok=True)
        return self

def get_data_paths() -> DataPaths:
    """
    Returns DataPaths configured to save data in the user's home directory.
    This fixes the issue where the base directory was set to a literal "~".
    """
    base = Path.home() / ".kaia"
    dp = DataPaths(base=base, models=base / "models",
                   logs=base / "logs", config=base / "config", bin=base / "bin",
                   temp=base / "temp", grammars=base / "grammars", templates=base / "templates",
                   docs=base / "docs")
    dp.ensure_dirs_exist()
    return dp
```

---
### File: src/killeraiagent/setup/__init__.py

```python

```

---
### File: src/killeraiagent/setup/setup_core.py

```python
"""
setup_core.py

Holds all shared internal functions for building llama.cpp, installing PyTorch,
voice, STT, GUI, teapot.ai, and downloading Gemma.

No interactive or wizard logic is present here.
"""

import os
import sys
import logging
import subprocess
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Optional

from killeraiagent.hardware import (
    detect_hardware_capabilities,
    get_optimal_llama_cpp_config,
    AcceleratorType
)
from killeraiagent.paths import get_data_paths
logger = logging.getLogger(__name__)

def run_subprocess(cmd: List[str],
                   check: bool = True,
                   env: Optional[Dict[str, str]] = None,
                   cwd: Optional[str] = None) -> Dict[str, str]:
    logging.info("Running command: " + " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=cwd
        )
        if check and result.returncode != 0:
            logging.error(f"Command failed with code {result.returncode}")
            logging.error(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": str(result.returncode)}
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return {"stdout": "", "stderr": str(e), "returncode": "-1"}

def log_documentation_paths():
    """
    Log the absolute paths to the project documentation (man pages)
    so users can easily navigate using clickable links (e.g. in VSCode).
    Assumes documentation is stored in a 'docs' folder at the project root.
    """
    project_root = Path(__file__).parent.parent.parent  # adjust if necessary
    docs_dir = project_root / "docs"
    if docs_dir.exists():
        for doc_file in docs_dir.glob("*.md"):
            logger.info(f"Documentation available: {doc_file.resolve()}")
    else:
        logger.info("No documentation folder found at expected location.")

def build_llama_cpp() -> bool:
    """
    Build llama.cpp from source with optimal configuration,
    copying binaries to ~/.kaia/bin.
    """
    logging.info("=== Building llama.cpp with acceleration support ===")
    hw_capabilities = detect_hardware_capabilities()
    config = get_optimal_llama_cpp_config(hw_capabilities)
    logging.info(f"Using {config.acceleration.value} acceleration")
    build_dir = tempfile.mkdtemp(prefix="llama_cpp_build_")
    logging.info(f"Temporary build directory: {build_dir}")

    try:
        git_cmd = ["git", "clone", "https://github.com/ggml-org/llama.cpp", build_dir]
        run_subprocess(git_cmd, check=True)

        # Use CMake's -S and -B flags to specify source and build dirs
        build_subdir = os.path.join(build_dir, "build")
        os.makedirs(build_subdir, exist_ok=True)
        
        # Basic CMake command
        cmake_cmd = ["cmake", f"-S{build_dir}", f"-B{build_subdir}"]
        
        # Add acceleration backend flags based on detected hardware
        if config.acceleration == AcceleratorType.CUDA and hw_capabilities.cuda.available:
            cmake_cmd.append("-DGGML_CUDA=ON")
            # Optional: Add compute capabilities if needed
            if hw_capabilities.cuda.compute_capability:
                cmake_cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={hw_capabilities.cuda.compute_capability}")
        
        elif config.acceleration == AcceleratorType.METAL:
            cmake_cmd.append("-DGGML_METAL=ON")
        
        elif config.acceleration == AcceleratorType.ROCM and hw_capabilities.rocm.available:
            cmake_cmd.append("-DGGML_HIP=ON")
            # Add AMD GPU targets if available
            if hw_capabilities.rocm.architectures:
                cmake_cmd.append(f"-DAMDGPU_TARGETS={';'.join(hw_capabilities.rocm.architectures)}")
        
        elif config.acceleration == AcceleratorType.VULKAN:
            cmake_cmd.append("-DGGML_VULKAN=ON")
        
        elif config.acceleration == AcceleratorType.OPENCL:
            cmake_cmd.append("-DGGML_OPENCL=ON")
            
        elif config.acceleration == AcceleratorType.SYCL:
            cmake_cmd.append("-DGGML_SYCL=ON")
            
        elif config.acceleration == AcceleratorType.MUSA:
            cmake_cmd.append("-DGGML_MUSA=ON")
            
        elif config.acceleration == AcceleratorType.CANN:
            cmake_cmd.append("-DGGML_CANN=ON")
            
        # Check for BLAS support (via environment variable)
        if os.environ.get("KAIA_USE_BLAS", "").lower() in ("1", "true", "yes"):
            cmake_cmd.append("-DGGML_BLAS=ON")
            blas_vendor = os.environ.get("KAIA_BLAS_VENDOR", "")
            if blas_vendor:
                cmake_cmd.append(f"-DGGML_BLAS_VENDOR={blas_vendor}")
        
        # Check for Arm KleidiAI support
        if hw_capabilities.cpu.supports_kleidiai:
            cmake_cmd.append("-DGGML_CPU_KLEIDIAI=ON")
        
        # Set build type
        cmake_cmd.append(f"-DCMAKE_BUILD_TYPE={config.build_type}")
        
        # Environment variables for the build
        build_env = os.environ.copy()
        env_vars_str = {k: str(v) for k, v in config.env_vars.items()}
        build_env.update(env_vars_str)
        
        # Run CMake configure
        run_subprocess(cmake_cmd, env=build_env, check=True)

        # Run CMake build
        build_cmd = ["cmake", "--build", build_subdir, "--config", config.build_type]
        run_subprocess(build_cmd, env=build_env, check=True)

        paths = get_data_paths()
        bin_dir = paths.bin
        bin_dir.mkdir(exist_ok=True)

        success = False
        binaries = ["llama-cli", "llama-server"]
        extensions = [".exe"] if os.name == "nt" else [""]
        search_dirs = [
            os.path.join(build_subdir, "bin"),
            os.path.join(build_subdir, "Release"),
            build_subdir,
            build_dir
        ]

        for binary in binaries:
            source_path = None
            for sd in search_dirs:
                if not os.path.exists(sd):
                    continue
                for ext in extensions:
                    guess = os.path.join(sd, binary + ext)
                    if os.path.exists(guess):
                        source_path = guess
                        break
                if source_path:
                    break
            if source_path:
                basename = os.path.basename(source_path)
                name_parts = os.path.splitext(basename)
                acc_suffix = f"-{config.acceleration.value}" if config.acceleration != AcceleratorType.CPU else ""
                target_name = f"{name_parts[0]}{acc_suffix}{name_parts[1]}"
                target_path = bin_dir / target_name
                shutil.copy2(source_path, target_path)
                logging.info(f"Copied {basename} to {target_path.resolve()}")
                success = True

        if success:
            logging.info("llama.cpp built successfully.")
            logging.info(f"Binaries are located in: {bin_dir.resolve()}")
            # Also log documentation paths for further user guidance
            log_documentation_paths()
            return True
        else:
            logging.error("Failed to find llama.cpp binaries after build.")
            return False
    except Exception as e:
        logging.error(f"Error building llama.cpp: {e}")
        logging.error(traceback.format_exc())
        return False
    finally:
        if os.environ.get("KAIA_KEEP_BUILD", "").lower() not in ("1", "true", "yes"):
            try:
                shutil.rmtree(build_dir)
                logging.info(f"Removed temporary build directory: {build_dir}")
            except Exception as e:
                logging.warning(f"Could not remove build directory: {e}")

def install_torch_with_acceleration() -> bool:
    from killeraiagent.hardware import detect_hardware_capabilities
    hw = detect_hardware_capabilities()
    acc = hw.primary_acceleration
    logging.info(f"Installing PyTorch for {acc.value}...")
    if acc == AcceleratorType.METAL:
        cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "torch", "torchvision", "torchaudio"]
    elif acc == AcceleratorType.CUDA and hw.cuda.available:
        cuver = hw.cuda.cuda_wheel_version or "cu121"
        cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir",
               f"--index-url=https://download.pytorch.org/whl/{cuver}",
               "torch", "torchvision", "torchaudio"]
    elif acc == AcceleratorType.ROCM and hw.rocm.available:
        cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir",
               "--index-url=https://download.pytorch.org/whl/rocm5.6",
               "torch", "torchvision", "torchaudio"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "torch", "torchvision", "torchaudio"]
    out = run_subprocess(cmd, check=True)
    return (out["returncode"] == "0")

def install_voice_dependencies() -> bool:
    logging.info("Installing voice dependencies (kokoro, sounddevice, soundfile)...")
    cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir",
           "kokoro", "sounddevice", "soundfile"]
    out = run_subprocess(cmd, check=True)
    return (out["returncode"] == "0")

def install_stt_dependencies() -> bool:
    logging.info("Installing STT dependencies (sherpa-onnx, torchaudio, gradio, numpy)...")
    cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir",
           "sherpa-onnx", "torchaudio", "gradio", "numpy"]
    out = run_subprocess(cmd, check=True)
    if out["returncode"] == "0":
        try:
            run_subprocess([
                sys.executable, "-c",
                "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Banafo/Kroko-ASR', repo_type='model')"
            ], check=True)
        except Exception as e:
            logging.error(f"Kroko-ASR model download error: {e}")
        return True
    return False

def install_gui_dependencies() -> bool:
    logging.info("Installing GUI dependencies (PySide6)...")
    cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "PySide6"]
    out = run_subprocess(cmd, check=True)
    return (out["returncode"] == "0")

def install_teapot_ai() -> bool:
    logging.info("Installing teapot.ai ...")
    cmd = [sys.executable, "-m", "pip", "install", "-U", "teapotai"]
    out = run_subprocess(cmd, check=True)
    return (out["returncode"] == "0")

def download_gemma_small() -> bool:
    from huggingface_hub import hf_hub_download
    model_repo = "lmstudio-community/gemma-3-1b-it-GGUF"
    filename = "gemma-3-1b-it-Q4_K_M.gguf"
    logging.info(f"Downloading {filename} from {model_repo} ...")
    try:
        paths = get_data_paths()
        model_dir = paths.models / model_repo.replace("/", "__")
        model_dir.mkdir(parents=True, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=model_repo,
            filename=filename,
            cache_dir=str(model_dir),
            force_download=False
        )
        logging.info(f"Downloaded model to: {local_path}")
        return True
    except Exception as e:
        logging.error(f"Error downloading gemma: {e}")
        return False
```

---
### File: src/killeraiagent/setup/setup_utils.py

```python
# killeraiagent/setup/setup_utils.py

import os
import sys
import logging
import datetime
import argparse
import shutil
from pathlib import Path

from killeraiagent.hardware import detect_hardware_capabilities, AcceleratorType
from killeraiagent.setup.setup_wizard import run_wizard
from killeraiagent.setup.setup_core import (
    build_llama_cpp,
    install_torch_with_acceleration,
    install_voice_dependencies,
    install_stt_dependencies,
    install_gui_dependencies
)
from killeraiagent.paths import get_data_paths

def is_dev_mode() -> bool:
    return os.environ.get("KAIA_DEV_MODE", "").lower() in ("1", "true", "yes")

def setup_logger() -> None:
    paths = get_data_paths()
    log_dir = paths.logs
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"kaia_{timestamp}.log"
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.info(f"Debug logs will be saved to: {log_file}")
    logging.info(f"Running in {'development' if is_dev_mode() else 'production'} mode")

def parse_args():
    parser = argparse.ArgumentParser(description="KAIA Setup Utilities")
    parser.add_argument("--all", action="store_true", help="Install all components")
    parser.add_argument("--torch", action="store_true", help="Install PyTorch with appropriate acceleration")
    parser.add_argument("--llama-cpp", action="store_true", help="Build llama.cpp with acceleration")
    parser.add_argument("--voice", action="store_true", help="Install voice support dependencies")
    parser.add_argument("--stt", action="store_true", help="Install STT support dependencies")
    parser.add_argument("--gui", action="store_true", help="Install GUI (PySide6) support")
    parser.add_argument("--acceleration", choices=["auto"] + [a.value for a in AcceleratorType],
                        default="auto", help="Acceleration type to use")
    parser.add_argument("--llama-cpp-path", type=str, help="Path to existing llama.cpp executable")
    return parser.parse_args()

def main():
    setup_logger()
    args = parse_args()
    logging.info("KAIA Setup Utilities")
    
    # If no flags provided, run the interactive wizard.
    if not any([
        args.all, args.torch, args.llama_cpp,
        args.voice, args.stt, args.gui, args.llama_cpp_path
    ]):
        logging.info("No flags detected, entering wizard mode.")
        run_wizard()
        return 0

    # Otherwise proceed with the legacy CLI installation process.
    if args.acceleration != "auto":
        os.environ["KAIA_ACCELERATION"] = args.acceleration

    hw_caps = detect_hardware_capabilities()
    logging.info(f"Detected primary accelerator: {hw_caps.primary_acceleration.value}")
    available = ", ".join(a.value for a in hw_caps.available_accelerators)
    logging.info(f"Available accelerators: {available}")

    if args.llama_cpp_path:
        cpp_path = Path(args.llama_cpp_path)
        if not cpp_path.exists():
            logging.error(f"Specified llama.cpp executable not found: {cpp_path}")
            return 1
        paths = get_data_paths()
        target_path = paths.bin / cpp_path.name
        shutil.copy2(cpp_path, target_path)
        logging.info(f"Copied {cpp_path.name} to {target_path}")

    success = True
    if args.all or args.torch:
        success = install_torch_with_acceleration() and success
    if args.all or args.llama_cpp and not args.llama_cpp_path:
        success = build_llama_cpp() and success
    if args.all or args.voice:
        success = install_voice_dependencies() and success
    if args.all or args.stt:
        success = install_stt_dependencies() and success
    if args.all or args.gui:
        success = install_gui_dependencies() and success

    if success:
        logging.info("Setup completed successfully!")
    else:
        logging.warning("Setup completed with some failures. Check the logs for details.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
```

---
### File: src/killeraiagent/setup/setup_wizard.py

```python
"""
setup_wizard.py

Interactive wizard for KAIA setup that guides the user through installation steps.
At each prompt you may answer 'y' (yes), 'n' (no), or 'help' for additional details.
A comprehensive FAQ document (in JSON Lines format) is expected to reside in the user's
data directory (~/.kaia/faq/kaia_faq.jsonl); if not present, it is copied from the project 
resources directory.
"""

import logging
import shutil
import importlib.util
from pathlib import Path

from killeraiagent.hardware import detect_hardware_capabilities
from killeraiagent.setup.setup_core import (
    install_torch_with_acceleration,
    build_llama_cpp,
    get_data_paths,
)
from killeraiagent.multiturn_teapot import MultiturnTeapot

logger = logging.getLogger(__name__)

def is_module_installed(module_name: str) -> bool:
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def prompt_with_help(prompt_text: str) -> str:
    """
    Prompt the user, accepting 'y', 'n', or 'help'. If the user answers 'help',
    an interactive help chat is launched for that topic.
    
    Returns "y" or "n".
    """
    while True:
        answer = input(f"{prompt_text} (y/n/help): ").strip().lower()
        if answer in ("y", "yes", "n", "no"):
            return "y" if answer in ("y", "yes") else "n"
        elif answer == "help":
            launch_help_chat(prompt_text)
            continue
        else:
            print("Please answer 'y', 'n', or 'help'.")

def launch_help_chat(topic: str) -> None:
    """
    Launch an interactive help chat about the given topic.
    """
    print(f"\n--- Help Session for: {topic} ---")
    assistant = MultiturnTeapot()
    assistant.rag_context = f"Help requested on topic: {topic}."
    print("Type your question about this step; type 'exit' to finish the help session.\n")
    while True:
        user_input = input("Help> ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Exiting help session.\n")
            break
        response = assistant.chat(user_input)
        print(f"Assistant: {response}\n")
    assistant.close()

def copy_faq_file_if_needed():
    """
    Copies the FAQ file from the project resources (resources/faq/kaia_faq.jsonl)
    to the user's data directory (~/.kaia/faq/kaia_faq.jsonl) if it does not exist.
    """
    paths = get_data_paths()
    faq_dir = paths.base / "faq"
    faq_dir.mkdir(parents=True, exist_ok=True)
    local_faq_path = faq_dir / "kaia_faq.jsonl"
    
    project_root = __file__.split("src/killeraiagent")[0]
    resource_faq_path = Path(project_root) / "resources" / "faq" / "kaia_faq.jsonl"
    
    if not local_faq_path.exists():
        try:
            shutil.copy2(str(resource_faq_path), str(local_faq_path))
            logger.info(f"Copied FAQ from {resource_faq_path} to {local_faq_path}")
        except Exception as e:
            logger.warning(f"Unable to copy FAQ from resources: {e}")

def is_llama_cpp_built() -> bool:
    """
    Check if local llama.cpp appears built by looking in the bin directory.
    If at least one binary (llama-cli or llama-server) exists, assume it is built.
    """
    paths = get_data_paths()
    bin_dir = paths.bin
    for binary in ["llama-cli", "llama-server"]:
        if any(bin_dir.glob(binary + "*")):
            return True
    return False

def run_wizard():
    print("--------------------------------------------------")
    print("Welcome to the KAIA Setup Wizard!")
    print("We will guide you step by step through the installation process.")
    print("At each prompt, answer 'y' (yes), 'n' (no), or 'help' for more details.\n")
    
    copy_faq_file_if_needed()
    
    hw = detect_hardware_capabilities()
    ram_gb = hw.memory.total_gb
    print(f"Detected {ram_gb:.1f} GB of RAM.\n")
    
    # Step 1: Check PyTorch installation.
    if is_module_installed("torch"):
        print("PyTorch is already installed with proper acceleration.\n")
    else:
        ans = prompt_with_help("Install PyTorch for hardware acceleration?")
        if ans == "y":
            if install_torch_with_acceleration():
                print("PyTorch installed successfully.\n")
            else:
                print("PyTorch installation failed. Check logs.\n")
        else:
            print("Skipping PyTorch installation.\n")
    
    # Step 2: Check teapot.ai installation.
    if is_module_installed("teapotai"):
        print("teapot.ai is already installed and will power on instantly.\n")
    else:
        ans = prompt_with_help("Install teapot.ai for easy QnA?")
        if ans == "y":
            from killeraiagent.setup.setup_core import install_teapot_ai
            if install_teapot_ai():
                print("teapot.ai installed successfully!\n")
            else:
                print("teapot.ai installation failed.\n")
        else:
            print("Skipping teapot.ai installation.\n")
    
    # Step 3: Offer to build local llama.cpp.
    if ram_gb >= 4:
        if is_llama_cpp_built():
            print("It appears that local llama.cpp is already built; skipping this step.\n")
            # Print the bin directory for user verification.
            bin_path = get_data_paths().bin.resolve()
            print(f"Built binaries are located here: {bin_path}\n")
        else:
            ans = prompt_with_help("Build llama.cpp for local inference?")
            if ans == "y":
                if build_llama_cpp():
                    print("llama.cpp built successfully.\n")
                    bin_path = get_data_paths().bin.resolve()
                    print(f"Binaries are located at: {bin_path}")
                    # Suggest opening the man page for further guidance.
                    project_root = Path(__file__).parent.parent.parent
                    doc_path = project_root / "docs" / "llamacpp_usage.md"
                    if doc_path.exists():
                        print(f"For detailed usage instructions, open: {doc_path.resolve()}\n")
                    else:
                        print("No detailed documentation found for llama.cpp usage.\n")
                    
                    ans2 = prompt_with_help("Download Gemma 3 1B (~800MB)?")
                    if ans2 == "y":
                        from killeraiagent.setup.setup_core import download_gemma_small
                        if download_gemma_small():
                            print("Gemma 3 1B downloaded successfully!\n")
                        else:
                            print("Gemma download failed.\n")
                else:
                    print("llama.cpp build failed.\n")
            else:
                print("Skipping local llama.cpp build.\n")
    else:
        print("System memory is less than 4GB; skipping local model build.\n")
    
    # Step 4: Offer to launch interactive help assistant.
    if is_module_installed("teapotai"):
        ans = prompt_with_help("Would you like to launch the interactive help assistant now?")
        if ans == "y":
            print("Launching interactive Teapot chat for help...\n")
            assistant = MultiturnTeapot()
            assistant.rag_context = ("Current status: PyTorch and teapot.ai are installed; "
                                     "check built binaries or documentation for details.")
            print("You can now ask questions about KAIA setup. For example, try 'doc llama' to get documentation on llama.cpp.")
            print("Type 'exit' to finish the help session.\n")
            while True:
                user_input = input("Help> ").strip()
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting help session.\n")
                    break
                response = assistant.chat(user_input)
                print(f"Assistant: {response}\n")
            assistant.close()
    else:
        print("teapot.ai not installed. Skipping help session.\n")
    
    print("Wizard complete! You can now run KAIA or import the library as needed.\n")

if __name__ == "__main__":
    run_wizard()
```

---
### File: src/voice/__init__.py

```python

```

---
### File: src/voice/api.py

```python
"""
High-level API for voice functionality.
"""
import os
import logging
from typing import Optional, Any, Tuple

from killeraiagent.features import features
from .io import play_audio, record_audio

logger = logging.getLogger(__name__)

def initialize_voice_support() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Initialize voice input and output support.
    
    Returns:
        Tuple of (voice_output, voice_input) objects or (None, None) if initialization failed
    """
    voice_output = None
    voice_input = None
    
    # Check for voice feature
    if not features.require("voice"):
        return None, None
        
    # Check for STT feature if needed
    use_stt = features.has("stt")
    
    try:
        # Import modules only after feature check
        from .engines.tts import KokoroTTS
        voice_output = KokoroTTS()
        logger.info("Voice output initialized successfully")
        
        if use_stt:
            from .engines.stt import KrokoSTT
            voice_input = KrokoSTT()
            logger.info("Voice input initialized successfully")
            
    except Exception as e:
        logger.warning(f"Failed to initialize voice support: {e}")
    
    return voice_output, voice_input

def text_to_speech(voice_output: Any, text: str) -> bool:
    """
    Convert text to speech and play it using the initialized TTS engine.
    
    Args:
        voice_output: Initialized TTS object
        text: Text to convert to speech
        
    Returns:
        True if successful, False otherwise
    """
    if voice_output is None:
        logger.debug("Voice output not available")
        return False
    
    try:
        # Generate audio from text
        waveform, sample_rate = voice_output.generate(text)
        
        # Play audio
        return play_audio(waveform, sample_rate)
    except Exception as e:
        logger.warning(f"Text-to-speech failed: {e}")
        return False

def speech_to_text(voice_input: Any) -> Optional[str]:
    """
    Convert speech to text using the initialized STT engine.
    
    Args:
        voice_input: Initialized STT object
        
    Returns:
        Transcribed text if successful, None otherwise
    """
    if voice_input is None:
        logger.debug("Voice input not available")
        return None
    
    try:
        # Record audio
        _, tmp_file = record_audio()
        if not tmp_file:
            return None
        
        try:
            # Transcribe the temporary file
            return voice_input.transcribe_file(tmp_file)
        finally:
            # Clean up temporary file
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {e}")
    
    except Exception as e:
        logger.warning(f"Speech-to-text failed: {e}")
        return None

def speak(text: str) -> bool:
    """
    Convenience function to speak text. Initializes voice support if needed.
    
    Args:
        text: Text to speak
        
    Returns:
        True if successful, False otherwise
    """
    # Initialize voice support if not already done
    voice_output, _ = initialize_voice_support()
    if voice_output is None:
        return False
    
    # Speak the text
    return text_to_speech(voice_output, text)

def listen() -> Optional[str]:
    """
    Convenience function to listen for speech. Initializes voice support if needed.
    
    Returns:
        Transcribed text if successful, None otherwise
    """
    # Initialize voice support if not already done
    _, voice_input = initialize_voice_support()
    if voice_input is None:
        return None
    
    # Listen and return transcribed text
    return speech_to_text(voice_input)
```

---
### File: src/voice/engines/__init__.py

```python

```

---
### File: src/voice/engines/stt.py

```python
"""
Speech-to-text engine implementation using Sherpa-ONNX.
"""
import os
import logging
import numpy as np
import torch
import torchaudio

from sherpa_onnx import OnlineRecognizer

logger = logging.getLogger(__name__)

class KrokoSTT:
    """
    Basic speech-to-text using Sherpa-ONNX. 
    We load multiple language models if needed, or just one (English, etc.).
    Example usage:
        stt = KrokoSTT("en")
        text = stt.transcribe_file("audio.wav")
    """

    def __init__(
        self,
        language:str = "en",
        tokens_path:str = "en_tokens.txt",
        encoder_path:str = "en_encoder.onnx",
        decoder_path:str = "en_decoder.onnx",
        joiner_path:str = "en_joiner.onnx"
    ):
        """
        Args:
            language: 'en', 'fr', 'de', 'es', ...
            tokens_path, encoder_path, decoder_path, joiner_path:
                paths to the model files for your chosen language.
        """
        self.language = language
        self.recognizer = OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,
            decoding_method="modified_beam_search",
            debug=False
        )
        logger.info(f"Initialized Sherpa-ONNX for language={language}")

    def transcribe_file(self, filepath:str) -> str:
        """
        Transcribes a whole audio file (blocking) and returns the text.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)
        waveform, sr = torchaudio.load(filepath)
        # ensure 16k
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            sr = 16000

        # convert to 1D if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        s = self.recognizer.create_stream()

        # chunk approach
        chunk_size = 3200  # ~0.2 seconds at 16k
        data = waveform.squeeze(0).numpy()
        offset = 0
        length = data.shape[0]
        while offset < length:
            end = offset + chunk_size
            chunk = data[offset:end]
            s.accept_waveform(sr, chunk.tolist())
            # decode partial
            while self.recognizer.is_ready(s):
                self.recognizer.decode_streams([s])
            offset = end

        # flush
        s.accept_waveform(sr, [])
        s.input_finished()
        while self.recognizer.is_ready(s):
            self.recognizer.decode_streams([s])

        result = self.recognizer.get_result(s)
        # handle type of result
        if isinstance(result, (list, np.ndarray)):
            result = " ".join(map(str, result))
        elif isinstance(result, bytes):
            result = result.decode("utf-8", errors="ignore")

        return result
```

---
### File: src/voice/engines/tts.py

```python
"""
Text-to-speech engine implementation using Kokoro.
"""
import logging
import numpy as np
import torch

from kokoro import KModel, KPipeline

logger = logging.getLogger(__name__)

class KokoroTTS:
    """
    Simple text-to-speech using Kokoro.
    Basic usage:
        tts = KokoroTTS("af_heart")  # load voice
        wav_samples, sr = tts.generate("Hello world")
        # write to .wav or play in memory
    """
    def __init__(self, voice:str = "af_heart", device:str = "cuda"):
        """
        Args:
            voice: voice name (like 'af_heart', 'am_michael', etc.)
            device: "cuda" or "cpu", if cuda is available
        """
        # We assume a single KModel is enough for all voices.
        # Alternatively you can load multiple if needed.
        self.model = KModel().to(device).eval()
        self.voice = voice
        self.device = device
        # create pipeline (lang_code='a' or 'b' based on voice name)
        if voice.startswith("af_") or voice.startswith("am_"):
            self.lang_code = "a"
        else:
            self.lang_code = "b"
        self.pipeline = KPipeline(lang_code=self.lang_code, model=False)
        # load voice data
        self.pack = self.pipeline.load_voice(voice)

    def generate(self, text:str, speed:float =1.0) -> tuple[np.ndarray, int]:
        """
        Convert text to audio. Returns (waveform, sample_rate).
        """
        # Tokenize text via pipeline
        phoneme_seq = []
        for _, ps, _ in self.pipeline(text, self.voice, speed):
            phoneme_seq = ps  # last chunk
        if not phoneme_seq:
            # maybe empty or error
            logger.warning("No phoneme sequence generated.")
            return (np.zeros(1, dtype=np.float32), 24000)

        ref_s = self.pack[len(phoneme_seq) - 1] if (len(phoneme_seq) > 0) else 0
        # Generate audio
        try:
            with torch.no_grad():
                if self.device == "cuda" and torch.cuda.is_available():
                    audio_t = self.model(phoneme_seq, ref_s, speed)
                else:
                    audio_t = self.model(phoneme_seq, ref_s, speed)
            # Convert to numpy
            wav = audio_t.cpu().numpy()
            return (wav, 24000)
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            return (np.zeros(1, dtype=np.float32), 24000)
```

---
### File: src/voice/io.py

```python
"""
Audio input/output utilities for voice functionality.
"""
import os
import logging
import tempfile
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def play_audio(waveform: np.ndarray, sample_rate: int) -> bool:
    """
    Play audio using available libraries.
    
    Args:
        waveform: Audio waveform as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        True if audio was played, False otherwise
    """
    try:
        import sounddevice as sd
        
        # Play the audio
        sd.play(waveform, sample_rate)
        sd.wait()  # Wait until audio is finished playing
        return True
    except ImportError:
        # Try alternative playback method
        return save_audio_temp(waveform, sample_rate)

def save_audio_temp(waveform: np.ndarray, sample_rate: int) -> bool:
    """
    Save audio to a temporary file when direct playback isn't available.
    
    Args:
        waveform: Audio waveform as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        True if audio was saved, False otherwise
    """
    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, waveform, sample_rate)
            logger.info(f"Audio saved to temporary file: {tmp.name}")
        return True
    except ImportError:
        logger.warning("Neither sounddevice nor soundfile available for audio playback")
        return False

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Record audio to a temporary file.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate for recording
        
    Returns:
        Tuple of (recorded_audio, temp_file_path) or (None, None) on failure
    """
    try:
        import sounddevice as sd
        import soundfile as sf
        
        # Create a temporary file for the recording
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        # Inform user
        logger.info(f"Recording for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete
        
        # Save to temporary file
        sf.write(tmp_file, recording, sample_rate)
        
        return recording, tmp_file
    except Exception as e:
        logger.warning(f"Error during recording: {e}")
        return None, None
```

---
### File: src/voice/speech_audio.py

```python
"""
src/killeraiagent/speech_audio.py

Speech/Audio adapter for KAIA that provides:
- Kokoro-based TTS (text -> audio)
- Sherpa-ONNX-based STT (audio -> text)

Requires 'kokoro', 'sherpa-onnx', 'sounddevice', 'soundfile', etc.
"""

import os
import numpy as np
import torch
import torchaudio
import logging

from kokoro import KModel, KPipeline
from sherpa_onnx import OnlineRecognizer
logger = logging.getLogger(__name__)

class KokoroTTS:
    """
    Simple text-to-speech using Kokoro.
    Basic usage:
        tts = KokoroTTS("af_heart")  # load voice
        wav_samples, sr = tts.generate("Hello world")
        # write to .wav or play in memory
    """
    def __init__(self, voice:str = "af_heart", device:str = "cuda"):
        """
        Args:
            voice: voice name (like 'af_heart', 'am_michael', etc.)
            device: "cuda" or "cpu", if cuda is available
        """
        # We assume a single KModel is enough for all voices.
        # Alternatively you can load multiple if needed.
        self.model = KModel().to(device).eval()
        self.voice = voice
        self.device = device
        # create pipeline (lang_code='a' or 'b' based on voice name)
        # For example if voice starts with 'a' => pipeline('a') else 'b'
        # but your example code used an approach like: 'af_heart' => pipeline 'a'
        if voice.startswith("af_") or voice.startswith("am_"):
            self.lang_code = "a"
        else:
            self.lang_code = "b"
        self.pipeline = KPipeline(lang_code=self.lang_code, model=False)
        # load voice data
        self.pack = self.pipeline.load_voice(voice)

    def generate(self, text:str, speed:float =1.0) -> tuple[np.ndarray, int]:
        """
        Convert text to audio. Returns (waveform, sample_rate).
        """
        # Tokenize text via pipeline
        # We produce partial phoneme sequences in a single pass
        # Then pass to the model
        # This is a naive approach. For large text, you'd chunk it.

        phoneme_seq = []
        for _, ps, _ in self.pipeline(text, self.voice, speed):
            phoneme_seq = ps  # last chunk
        if not phoneme_seq:
            # maybe empty or error
            logger.warning("No phoneme sequence generated.")
            return (np.zeros(1, dtype=np.float32), 24000)

        ref_s = self.pack[len(phoneme_seq) - 1] if (len(phoneme_seq) > 0) else 0
        # Generate audio
        try:
            with torch.no_grad():
                if self.device == "cuda" and torch.cuda.is_available():
                    audio_t = self.model(phoneme_seq, ref_s, speed)
                else:
                    audio_t = self.model(phoneme_seq, ref_s, speed)
            # Convert to numpy
            wav = audio_t.cpu().numpy()
            return (wav, 24000)
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            return (np.zeros(1, dtype=np.float32), 24000)


class KrokoSTT:
    """
    Basic speech-to-text using Sherpa-ONNX. 
    We load multiple language models if needed, or just one (English, etc.).
    Example usage:
        stt = KrokoSTT("en")
        text = stt.transcribe_file("audio.wav")
    """

    def __init__(
        self,
        language:str = "en",
        tokens_path:str = "en_tokens.txt",
        encoder_path:str = "en_encoder.onnx",
        decoder_path:str = "en_decoder.onnx",
        joiner_path:str = "en_joiner.onnx"
    ):
        """
        Args:
            language: 'en', 'fr', 'de', 'es', ...
            tokens_path, encoder_path, decoder_path, joiner_path:
                paths to the model files for your chosen language.
        """
        self.language = language
        self.recognizer = OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,
            decoding_method="modified_beam_search",
            debug=False
        )
        logger.info(f"Initialized Sherpa-ONNX for language={language}")

    def transcribe_file(self, filepath:str) -> str:
        """
        Transcribes a whole audio file (blocking) and returns the text.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)
        waveform, sr = torchaudio.load(filepath)
        # ensure 16k
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            sr = 16000

        # convert to 1D if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        s = self.recognizer.create_stream()

        # chunk approach
        chunk_size = 3200  # ~0.2 seconds at 16k
        data = waveform.squeeze(0).numpy()
        offset = 0
        length = data.shape[0]
        while offset < length:
            end = offset + chunk_size
            chunk = data[offset:end]
            s.accept_waveform(sr, chunk.tolist())
            # decode partial
            while self.recognizer.is_ready(s):
                self.recognizer.decode_streams([s])
            offset = end

        # flush
        s.accept_waveform(sr, [])
        s.input_finished()
        while self.recognizer.is_ready(s):
            self.recognizer.decode_streams([s])

        result = self.recognizer.get_result(s)
        # handle type of result
        if isinstance(result, (list, np.ndarray)):
            result = " ".join(map(str, result))
        elif isinstance(result, bytes):
            result = result.decode("utf-8", errors="ignore")

        return result
```

---

