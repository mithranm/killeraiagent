"""
Tests for the redesigned LLM model module.
Developers can create LLM instances via the create_model factory.
"""

import pytest
from unittest.mock import patch
from killeraiagent.model import ModelInfo, create_model, LlamaCppModel, HuggingFaceModel

def test_model_info():
    """Test ModelInfo's to_dict and from_dict conversions."""
    info = ModelInfo(
        model_id="test-model",
        model_path="/path/to/model",
        model_type="llamacpp",
        context_length=4096,
        requires_gpu=False,
        model_size_gb=1.0,
        description="Test model",
        quantization="Q4_K_M"
    )
    data = info.to_dict()
    assert data["model_id"] == "test-model"
    info2 = ModelInfo.from_dict(data)
    assert info2.model_id == info.model_id
    assert info2.model_path == info.model_path

@patch("killeraiagent.model.LlamaCppModel.load", return_value=True)
@patch("killeraiagent.model.LlamaCppModel.generate")
def test_create_model_llamacpp(mock_generate, mock_load):
    """
    Test creation of a LlamaCppModel instance using the factory.
    Verify that custom chat_template is passed and that the generate method is called.
    """
    # Create a llama-cpp instance with a custom chat template
    llm = create_model(
        model_path="/fake/path/to/llama_model.bin",
        backend="llamacpp",
        chat_template="custom_template",
        context_length=4096,
        n_threads=2  # example extra parameter
    )
    assert isinstance(llm, LlamaCppModel)
    # Ensure the custom chat_template is set
    assert llm.chat_template == "custom_template"
    
    # Patch generate to simulate a response
    mock_generate.return_value = ("mock response", {"raw": "data"})
    text, meta = llm.generate("Hello")
    assert text == "mock response"

@patch("killeraiagent.model.HuggingFaceModel.load", return_value=True)
@patch("killeraiagent.model.HuggingFaceModel.generate")
def test_create_model_hf(mock_generate, mock_load):
    """
    Test creation of a HuggingFaceModel instance using the factory.
    Verify that context_length and custom chat_template are passed to the instance.
    """
    llm = create_model(
        model_path="facebook/opt-350m",
        backend="hf",
        chat_template="my_chat_template",
        context_length=2048,
        verbose=True
    )
    assert isinstance(llm, HuggingFaceModel)
    # Verify that the context length is stored in the model info
    assert llm.model_info.context_length == 2048
    # Verify that the chat_template is passed (if provided)
    assert llm.chat_template == "my_chat_template"
    
    mock_generate.return_value = ("hf response", {"raw": "data"})
    text, meta = llm.generate("Hi")
    assert text == "hf response"

def test_invalid_backend():
    """Test that an invalid backend string raises a ValueError."""
    with pytest.raises(ValueError):
        create_model(model_path="some_path", backend="invalid")
