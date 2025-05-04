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