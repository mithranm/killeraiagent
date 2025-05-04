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