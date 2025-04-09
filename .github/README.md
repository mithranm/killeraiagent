# KAIA - Killer AI Agents

[![PyPI version](https://badge.fury.io/py/killeraiagent.svg)](https://badge.fury.io/py/killeraiagent)
[![Test Package](https://github.com/username/killeraiagent/actions/workflows/test.yml/badge.svg)](https://github.com/username/killeraiagent/actions/workflows/test.yml)
[![Documentation Status](https://github.com/username/killeraiagent/actions/workflows/docs.yml/badge.svg)](https://username.github.io/killeraiagent/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

KAIA (Killer AI Agents) is a modular, extensible platform for creating, managing, and interacting with AI agents that can perform tasks on a user's computer.

## Features

- **Universal LLM Interface**: Support for multiple LLM engines including llama-cpp-python and Hugging Face Transformers
- **Hardware-Aware**: Auto-detects system capabilities and optimizes LLM performance
- **Resource Management**: Intelligent allocation of CPU, memory, and GPU resources
- **Integration**: Designed to work with tools like LlamaSearch by providing a unified LLM interface
- **Extensible**: Modular design allows for adding new capabilities and tools

## Installation

```bash
pip install killeraiagent
```

For development installation:

```bash
git clone https://github.com/username/killeraiagent.git
cd killeraiagent
pip install -e .
```

## Basic Usage

```python
from killeraiagent import get_kaia

# Get the KAIA instance with automatic hardware optimization
kaia = get_kaia(auto_optimize=True)

# Load a model (will use recommended model based on hardware if none specified)
kaia.load_model()

# Generate text
text, metadata = kaia.generate(
    prompt="Explain the concept of artificial intelligence.",
    max_tokens=512,
    temperature=0.7
)

print(text)
```

## LlamaSearch Integration

KAIA is designed to work seamlessly with LlamaSearch:

```python
from llamasearch.core.llm import LLMSearch
from killeraiagent import get_kaia

# Initialize KAIA and load a model
kaia = get_kaia()
model = kaia.load_model("qwen2.5-1.5b-instruct")

# Initialize LlamaSearch with KAIA's model
llm_search = LLMSearch(
    storage_dir="/path/to/storage",
    models_dir="/path/to/models",
    external_model=model  # Pass the model instance from KAIA
)

# Use LlamaSearch's RAG capabilities with KAIA's model
response = llm_search.llm_query("What information do we have about neural networks?")
print(response["response"])
```

## Hardware Optimization

KAIA automatically detects and optimizes for your hardware:

```python
from killeraiagent import get_kaia

kaia = get_kaia()

# Get hardware information
hw_info = kaia.get_hardware_info()
print(f"CPU: {hw_info['cpu']['physical_cores']} cores, {hw_info['cpu']['total_memory_gb']} GB RAM")
if hw_info['gpu']['has_cuda']:
    print(f"GPU: {hw_info['gpu']['devices'][0]['name']}")

# Get optimal configuration
optimal_config = kaia.get_optimal_config()
print(f"Recommended model: {optimal_config['recommended_model']['model_id']}")
print(f"GPU layers: {optimal_config['llm']['n_gpu_layers']}")
```

## Documentation

Full documentation is available at [https://username.github.io/killeraiagent/](https://username.github.io/killeraiagent/)

## Testing

Run the test suite:

```bash
pytest
```

## License

Apache License 2.0

## Author

Mithran Mohanraj (mithran.mohanraj@gmail.com)