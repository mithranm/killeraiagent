"""
Tests for the main KAIA module.
This new design lets developers create independent KAIA instances via get_kaia,
which wraps an underlying model instance (created by create_model).
"""

import pytest
from unittest.mock import patch

from killeraiagent.kaia import KAIA, get_kaia
from killeraiagent.model import HuggingFaceModel, LlamaCppModel

def test_kaia_instance_creation_hf():
    """
    Test that get_kaia returns a KAIA instance with a HuggingFace backend
    when backend is set to 'hf'.
    """
    # For this test we simulate creation (the underlying model won't really load)
    kaia_instance = get_kaia(model_path="facebook/opt-350m", backend="hf", context_length=2048)
    assert isinstance(kaia_instance, KAIA)
    # The underlying model should be an instance of HuggingFaceModel
    from killeraiagent.model import HuggingFaceModel
    assert isinstance(kaia_instance.model, HuggingFaceModel)
    # Also verify that the model info includes the correct context length.
    assert kaia_instance.model.model_info.context_length == 2048

def test_kaia_instance_creation_llamacpp():
    """
    Test that get_kaia returns a KAIA instance with a llama-cpp backend
    when backend is set to 'llamacpp'.
    """
    kaia_instance = get_kaia(model_path="/fake/path/to/llama_model.bin", backend="llamacpp", context_length=4096)
    assert isinstance(kaia_instance, KAIA)
    from killeraiagent.model import LlamaCppModel
    assert isinstance(kaia_instance.model, LlamaCppModel)
    assert kaia_instance.model.model_info.context_length == 4096

def test_kaia_generate(monkeypatch):
    """
    Test that the KAIA.generate method delegates to the underlying model.
    """
    # Create a dummy model that implements load and generate.
    class DummyModel:
        def __init__(self):
            self.model_info = type("dummy", (), {"model_id": "dummy-model"})
        def load(self):
            return True
        def generate(self, prompt, **kwargs):
            return "dummy generation", {"raw": "dummy"}
        def embed(self, text):
            return [0.1, 0.2, 0.3]
    
    # Create a KAIA instance that wraps the dummy model.
    dummy_instance = KAIA(DummyModel())
    # Test generate
    generated_text, meta = dummy_instance.generate("Hello, world!")
    assert generated_text == "dummy generation"
    assert meta["raw"] == "dummy"

def test_kaia_embed(monkeypatch):
    """
    Test that KAIA.embed properly delegates to the underlying model.
    """
    class DummyModel:
        def __init__(self):
            self.model_info = type("dummy", (), {"model_id": "dummy-model"})
        def load(self):
            return True
        def generate(self, prompt, **kwargs):
            return "dummy generation", {"raw": "dummy"}
        def embed(self, text):
            # Return a fixed vector
            return [0.5, 0.6, 0.7]
    
    dummy_instance = KAIA(DummyModel())
    embedding = dummy_instance.embed("Test input")
    assert embedding == [0.5, 0.6, 0.7]

def test_invalid_backend():
    """
    Test that get_kaia raises a ValueError when an invalid backend is supplied.
    """
    with pytest.raises(ValueError):
        get_kaia(model_path="some_model", backend="invalid")