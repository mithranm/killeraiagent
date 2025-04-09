# KAIA - Killer AI Agents

KAIA (Killer AI Agents) is a modular, extensible platform for creating, managing, and interacting with AI agents that can perform tasks on a user's computer.

## Features

- **Universal LLM Interface**: Support for multiple LLM engines including llama-cpp-python and Hugging Face Transformers
- **Hardware-Aware**: Auto-detects system capabilities and optimizes LLM performance
- **Resource Management**: Intelligent allocation of CPU, memory, and GPU resources
- **Integration**: Designed to work with tools like LlamaSearch by providing a unified LLM interface
- **Extensible**: Modular design allows for adding new capabilities and tools

## Quick Start

```bash
# Install from PyPI
pip install killeraiagent

# Use in Python
from killeraiagent import get_kaia

# Initialize KAIA with automatic hardware optimization
kaia = get_kaia(auto_optimize=True)

# Load a model (uses recommended model based on hardware if none specified)
kaia.load_model()

# Generate text
text, metadata = kaia.generate(
    prompt="Explain the concept of artificial intelligence.",
    max_tokens=512,
    temperature=0.7
)

print(text)
```

## Project Structure

- **killeraiagent.kaia**: Main entry point for the package
- **killeraiagent.model**: Universal interface for different LLM backends
- **killeraiagent.resources**: Hardware detection and resource management

## License

Apache License 2.0