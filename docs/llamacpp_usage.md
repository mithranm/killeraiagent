# LlamaCPP Usage in KAIA

## Overview

KAIA (Killer AI Agents) integrates with llama.cpp to provide optimized inference for LLaMa-based models on various hardware configurations. This document outlines how to use llama.cpp within the KAIA backend.

## Requirements

- Built llama.cpp binaries (automatically installed during KAIA setup)
- GGUF model files
- Hardware acceleration (optional but recommended)

## Model Selection

KAIA automatically selects the appropriate model based on your system's capabilities:

| Hardware Configuration | Recommended Model |
|------------------------|-------------------|
| High-end (16GB+ RAM, GPU) | Gemma 3 4B Instruct |
| Mid-range (8GB RAM) | Gemma 1B Instruct |
| Low-end (4GB RAM) | TeapotAI/TeapotLLM |

## Basic Usage

```python
from killeraiagent.models.llamacpp.llamacpp_subprocess import LlamaCppSubprocess
from killeraiagent.models.base import ModelInfo

# Create model info
model_info = ModelInfo(
    model_id="gemma-3-1b-it",
    model_engine="llamacpp",
    description="Gemma 3 1B Instruct GGUF model",
    context_length=4096,
    requires_gpu=False,
    model_size_gb=1.1
)

# Initialize the model (loads model through subprocess)
llm = LlamaCppSubprocess(
    model_info=model_info,
    model_path="~/.kaia/models/lmstudio-community__gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf",
    n_gpu_layers=0  # Set to higher value to use GPU acceleration
)

# Generate text
response, _ = llm.generate(
    prompt="Write a short poem about AI.",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1
)

print(response)
```

## CLI Usage

KAIA provides a convenient wrapper around the llama.cpp binaries:

```bash
kaia-cli chat --model gemma-3-1b-it
```

## GPU Acceleration

To enable GPU acceleration, use the `--n-gpu-layers` parameter:

```bash
kaia-cli chat --model gemma-3-1b-it --n-gpu-layers 33
```

The optimal value depends on your GPU and model size. For most consumer GPUs:

| GPU Memory | Recommended n-gpu-layers |
|------------|--------------------------|
| 4GB VRAM   | 20-24                    |
| 8GB VRAM   | 32-35                    |
| 16GB+ VRAM | All layers (50+)         |

## Supported Acceleration Backends

KAIA automatically builds llama.cpp with support for various acceleration backends:

- CUDA (NVIDIA GPUs)
- Metal (Apple Silicon/AMD GPUs on macOS)
- ROCm/HIP (AMD GPUs on Linux)
- Vulkan (Cross-platform)
- OpenCL (Various GPUs)
- SYCL (Intel GPUs)
- BLAS (CPU acceleration)
- Arm KleidiAI (Arm CPUs)

The appropriate backend is selected based on your hardware configuration.

## Advanced Parameters

Fine-tune generation with these parameters:

```python
llm.generate(
    prompt="Your prompt here",
    max_tokens=512,         # Maximum number of tokens to generate
    temperature=0.8,        # Higher values increase randomness (0.0-1.0)
    top_p=0.9,              # Nucleus sampling parameter (0.0-1.0)
    top_k=40,               # Limit vocabulary to top K options
    repeat_penalty=1.1,     # Penalize repetition (1.0 = no penalty)
    presence_penalty=0.0,   # Penalize new tokens based on presence in text
    frequency_penalty=0.0,  # Penalize new tokens based on frequency in text
    mirostat=0,             # Enable Mirostat sampling (0, 1, or 2)
    mirostat_tau=5.0,       # Mirostat target entropy
    mirostat_eta=0.1        # Mirostat learning rate
)
```

## Troubleshooting

Common issues and solutions:

1. **Out of memory errors**: Reduce context length, batch size, or try a smaller model
2. **Slow inference**: Enable GPU acceleration with `--n-gpu-layers` or increase thread count
3. **GPU not detected**: Ensure proper GPU drivers are installed and update to the latest version
4. **Crashes during generation**: Try reducing the number of GPU layers or using CPU-only mode

## Performance Optimization

To optimize performance:

1. Set appropriate thread count (`--threads`) based on your CPU cores
2. Enable GPU acceleration for larger models
3. Use quantized models (Q4_K_M, Q5_K_M) for better performance
4. Adjust batch size (`--batch-size`) for prompt processing

For more information, see the [KAIA documentation](https://github.com/teapotai/kaia/docs).