# KAIA Setup Guide

## Introduction

KAIA (Killer AI Agents) is a modular, extensible platform for creating, managing, and interacting with AI agents. This guide will help you install and configure KAIA on your system.

## System Requirements

### Minimum Requirements
- Python 3.11+
- 4GB RAM
- x86_64 or ARM64 CPU
- 2GB free disk space

### Recommended Requirements
- 8GB+ RAM
- Dedicated GPU/NPU with 4GB+ VRAM
- 10GB free disk space

### Optimal Requirements
- 16GB+ RAM
- CUDA-capable GPU or Apple Silicon
- 20GB free disk space

## Installation

### Basic Installation

```bash
# Create a virtual environment (recommended)
python -m venv kaia-env
source kaia-env/bin/activate  # On Windows: kaia-env\Scripts\activate

# Install KAIA from PyPI
pip install killeraiagent

# Initialize KAIA
kaia-setup init
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/teapotai/kaia.git
cd kaia

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Initialize KAIA
python -m killeraiagent.setup.setup_wizard
```

## Configuration

KAIA stores its configuration in `~/.kaia/config/kaia_config.json`. You can modify this file directly or use the setup wizard:

```bash
kaia-setup config
```

### Directory Structure

KAIA creates the following directory structure:

```
~/.kaia/
├── bin/          # Binary files (llama.cpp executables)
├── config/       # Configuration files
├── docs/         # Documentation
├── grammars/     # Grammar files for constrained generation
├── logs/         # Log files
├── models/       # Downloaded models
├── temp/         # Temporary files
└── templates/    # Templates for agents
```

## Model Setup

### Downloading Models

KAIA can automatically download models based on your system capabilities:

```bash
# Download recommended model for your hardware
kaia-setup download-model

# Download specific model
kaia-setup download-model --model gemma-1b
```

### Using Custom Models

Place your GGUF models in `~/.kaia/models/` or specify a custom path:

```bash
kaia-cli chat --model /path/to/your/model.gguf
```

## Acceleration Setup

KAIA automatically detects and uses available hardware acceleration:

### NVIDIA GPU (CUDA)

For NVIDIA GPUs, KAIA will automatically build llama.cpp with CUDA support. Ensure you have:
- NVIDIA GPU drivers installed
- CUDA toolkit installed (optional, but recommended)

KAIA will automatically detect your GPU capabilities and configure llama.cpp optimally.

### AMD GPU (ROCm/HIP)

For AMD GPUs on Linux, KAIA will build llama.cpp with HIP support if ROCm is installed:
- Install ROCm toolkit for your distribution
- Ensure rocm-smi is available in your PATH

KAIA will detect your AMD GPU architecture and use the appropriate settings.

### Apple Silicon / Mac (Metal)

On macOS with Apple Silicon or supported AMD GPUs, KAIA automatically uses Metal acceleration:
- No additional setup required
- Works best on Apple M1/M2/M3 series chips

### Intel GPU (SYCL)

For Intel GPUs, KAIA can build llama.cpp with SYCL support:
- Install the Intel oneAPI Base Toolkit
- Set the KAIA_USE_SYCL environment variable: `export KAIA_USE_SYCL=1`

### Cross-Platform GPU (Vulkan)

For systems without specific GPU vendor support, KAIA can use Vulkan:
- Install Vulkan SDK for your platform
- Ensure vulkaninfo is available in your PATH

### Arm-Based Systems (KleidiAI)

For Arm-based systems, KAIA can leverage Arm KleidiAI optimizations:
- Automatically enabled on supported Arm processors
- For SME acceleration: `export GGML_KLEIDIAI_SME=1`

## Optional Components

### Voice Support

Install voice synthesis and recognition:

```bash
kaia-setup install-voice
```

### GUI

Install graphical user interface:

```bash
kaia-setup install-gui
```

### Speech-to-Text

Install speech recognition:

```bash
kaia-setup install-stt
```

## Running KAIA

### Command Line

Start a chat session with a default model:

```bash
kaia-cli chat
```

Use a specific model with GPU acceleration:

```bash
kaia-cli chat --model gemma-3-1b-it --n-gpu-layers 33
```

### GUI (if installed)

```bash
kaia-gui
```

## Advanced Configuration

### Environment Variables

- `KAIA_CONFIG_PATH`: Override default config path
- `KAIA_MODELS_DIR`: Override default models directory
- `KAIA_LOG_LEVEL`: Set logging level (INFO, DEBUG, WARNING)
- `KAIA_KEEP_BUILD`: Keep build directories (set to 1 for debugging)
- `KAIA_USE_BLAS`: Enable BLAS acceleration (set to 1 to enable)
- `KAIA_BLAS_VENDOR`: Set BLAS vendor (OpenBLAS, Intel10_64lp, etc.)

### Custom Build Options

To customize llama.cpp build, create a `~/.kaia/config/build_confi