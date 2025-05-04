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
