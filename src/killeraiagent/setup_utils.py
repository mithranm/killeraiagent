#!/usr/bin/env python
"""
setup_utils.py - KillerAI Agent Setup Utilities

This module provides installation and configuration utilities for KillerAI Agent.
It includes functions for detecting acceleration (CUDA, Metal, or CPU), configuring the build
environment for llama-cpp-python, installing PyTorch with the proper backend support, and logging.
Additionally, a custom post-install command (PostInstallCommand) is defined that runs these
steps automatically after the package is installed.
"""

import os
import platform
import subprocess
import sys
import logging
import datetime
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

def is_dev_mode() -> bool:
    """Determine if running in development mode."""
    return os.environ.get("KAIA_DEV_MODE", "").lower() in ("1", "true", "yes")

def get_killeraiagent_dir() -> Path:
    """Return the base directory for KillerAI Agent data."""
    if is_dev_mode():
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                return parent
        return Path.cwd()
    else:
        base = os.path.join(os.path.expanduser("~"), ".killeraiagent")
        return Path(os.environ.get("KAIA_DATA_DIR", base))

def get_data_paths() -> Dict[str, Path]:
    """Return required data directories for KillerAI Agent."""
    base = get_killeraiagent_dir()
    paths = {
        "base": base,
        "models": base / "models",
        "logs": base / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

def get_setup_config_path() -> Path:
    """Return the path to the setup configuration file."""
    return get_killeraiagent_dir() / "setup_config.json"

def load_setup_config() -> Dict[str, Any]:
    """Load the setup configuration from a JSON file."""
    config_path = get_setup_config_path()
    if config_path.exists():
        try:
            with config_path.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {
        "setup_complete": False,
        "steps_completed": [],
        "llama_cpp_installed": False,
        "torch_installed": False,
        "last_setup_attempt": None
    }

def save_setup_config(config: Dict[str, Any]) -> None:
    """Save the setup configuration to a JSON file."""
    config["last_setup_attempt"] = datetime.datetime.now().isoformat()
    config_path = get_setup_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

def mark_step_complete(step: str) -> None:
    """Mark a setup step as complete in the configuration."""
    config = load_setup_config()
    if step not in config["steps_completed"]:
        config["steps_completed"].append(step)
    if step == "llama_cpp_install":
        config["llama_cpp_installed"] = True
    elif step == "torch_install":
        config["torch_installed"] = True
    all_steps = ["llama_cpp_install", "torch_install"]
    if all(s in config["steps_completed"] for s in all_steps):
        config["setup_complete"] = True
    save_setup_config(config)

def setup_logger() -> None:
    """Configure logging to output to both console and file."""
    paths = get_data_paths()
    log_dir = paths["logs"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"setup_debug_{timestamp}.log"
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Clear existing handlers
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

def run_subprocess(cmd: List[str], env: Optional[Dict[str, str]] = None,
                   cwd: Optional[str] = None, check: bool = True) -> Dict[str, Any]:
    """Run a subprocess command and return its output."""
    logging.info("Running command: " + " ".join(cmd))
    result = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        logging.error(f"Command failed with code {result.returncode}")
        logging.error(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

def detect_optimal_acceleration() -> Dict[str, Any]:
    """
    Detect the optimal acceleration configuration.
    Returns a dict with keys:
      - type: "cuda", "metal", or "cpu"
      - cuda_version, cuda_path, etc. (if applicable)
    """
    result = {"type": "cpu", "cuda_version": None, "cuda_path": None, "nvcc_path": None}
    sys_platform = platform.system()
    if sys_platform == "Windows":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and os.path.exists(cuda_path):
            result.update({"cuda_path": cuda_path, "type": "cuda"})
            nvcc = os.path.join(cuda_path, "bin", "nvcc.exe")
            if os.path.exists(nvcc):
                result["nvcc_path"] = nvcc
            result["cuda_version"] = "12.4"
    elif sys_platform == "Linux":
        try:
            proc = run_subprocess(["nvcc", "--version"], check=False)
            if proc["returncode"] == 0:
                result["type"] = "cuda"
                import re
                match = re.search(r"release (\d+\.\d+)", proc["stdout"])
                if match:
                    result["cuda_version"] = match.group(1)
        except Exception:
            pass
    elif sys_platform == "Darwin":
        if platform.machine() == "arm64":
            result["type"] = "metal"
        else:
            result["type"] = "cpu"
    return result

def map_cuda_version_to_wheel(version: str) -> Optional[str]:
    """Map a CUDA version string to the corresponding PyTorch wheel version."""
    version = version.strip()
    if version.startswith("12"):
        return "cu121"
    elif version.startswith("11.8"):
        return "cu118"
    elif version.startswith("11.7"):
        return "cu117"
    elif version.startswith("11"):
        return "cu116"
    return None

def get_acceleration_config() -> Dict[str, Any]:
    """Return acceleration configuration from environment or auto-detection."""
    accel = os.environ.get("KAIA_ACCELERATION", "auto").lower()
    if accel == "auto":
        return detect_optimal_acceleration()
    config = {"type": accel}
    if accel == "cuda":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            config["cuda_path"] = cuda_path
        cuda_version = os.environ.get("KAIA_CUDA_VERSION")
        if cuda_version:
            config["cuda_version"] = cuda_version
            wheel_ver = map_cuda_version_to_wheel(cuda_version)
            if wheel_ver:
                config["cuda_wheel_version"] = wheel_ver
    return config

def check_llama_cpp_cuda_support() -> bool:
    """
    Check if the installed llama-cpp-python package has CUDA support.
    This attempts several methods: checking the version string, loading the shared library,
    and trying to initialize a model with GPU layers.
    """
    try:
        result = run_subprocess([sys.executable, "-m", "pip", "show", "llama-cpp-python"], check=False)
        if result["returncode"] != 0:
            logging.info("llama-cpp-python not installed.")
            return False
        import llama_cpp  # type: ignore
        if hasattr(llama_cpp, "__version__") and "cuda" in llama_cpp.__version__.lower():
            logging.info("CUDA support detected via version string.")
            return True
        from llama_cpp import Llama  # type: ignore
        try:
            from llama_cpp._ctypes_extensions import load_shared_library
            import pathlib
            lib_path = pathlib.Path(llama_cpp.__file__).parent / "lib"
            lib = load_shared_library('llama', lib_path)
            if hasattr(lib, "llama_supports_gpu_offload") and callable(lib.llama_supports_gpu_offload):
                if bool(lib.llama_supports_gpu_offload()):
                    logging.info("CUDA support detected via shared library.")
                    return True
        except Exception as lib_err:
            logging.debug(f"Shared library check error: {lib_err}")
        try:
            _ = Llama(model_path="", n_gpu_layers=1)
            logging.info("CUDA support detected via Llama instantiation.")
            return True
        except Exception as model_err:
            if "cuda" in str(model_err).lower():
                logging.info("CUDA support detected via CUDA error message.")
                return True
        logging.info("No CUDA support detected in llama-cpp-python.")
        return False
    except Exception as e:
        logging.error(f"Error during CUDA support check: {e}")
        return False

def configure_build_environment(config: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Configure the build environment for building llama-cpp-python.
    Returns a tuple of (acceleration type, list of CMake arguments, environment variables).
    """
    cmake_args = []
    env_vars: Dict[str, str] = {}
    acc_type = config.get("type", "cpu")
    if acc_type == "metal":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            cmake_args.extend([
                "-DGGML_METAL=on",
                "-DCMAKE_OSX_ARCHITECTURES=arm64",
                "-DCMAKE_APPLE_SILICON_PROCESSOR=arm64"
            ])
            env_vars["FORCE_CMAKE"] = "1"
            logging.info("Configuring for Metal acceleration on Apple Silicon.")
        else:
            logging.warning("Metal acceleration requested but unsupported; defaulting to CPU.")
            acc_type = "cpu"
    elif acc_type == "cuda":
        env_vars["FORCE_CMAKE"] = "1"
        cmake_args.extend(["-DGGML_CUDA=on", "-DCMAKE_CUDA_ARCHITECTURES=all"])
        cuda_path = config.get("cuda_path")
        if cuda_path and os.path.exists(cuda_path):
            env_vars["CUDA_PATH"] = cuda_path
            nvcc = os.path.join(cuda_path, "bin", "nvcc.exe")
            if os.path.exists(nvcc):
                env_vars["CUDACXX"] = nvcc
            logging.info(f"Using NVCC at: {env_vars.get('CUDACXX')}")
    if cmake_args:
        env_vars["CMAKE_ARGS"] = " ".join(cmake_args)
    return acc_type, cmake_args, env_vars

def install_llama_cpp_python() -> bool:
    """Install llama-cpp-python from source with acceleration support."""
    LATEST_VERSION = "0.3.8"  # Update as needed
    logging.info("=== Installing llama-cpp-python with acceleration support ===")
    acc_config = get_acceleration_config()
    acc_type = acc_config.get("type", "cpu")
    if acc_type == "metal" and not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        logging.warning("Metal acceleration not supported on this platform; defaulting to CPU.")
        acc_type = "cpu"
        acc_config["type"] = "cpu"
    elif acc_type == "cuda" and platform.system() == "Darwin":
        logging.warning("CUDA not available on macOS; switching to Metal if on Apple Silicon, else CPU.")
        acc_type = "metal" if platform.machine() == "arm64" else "cpu"
        acc_config["type"] = acc_type
    logging.info(f"Detected acceleration type: {acc_type}")
    _, cmake_args, env_vars = configure_build_environment(acc_config)
    
    try:
        run_subprocess([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"], check=False)
        logging.info("Uninstalled any existing llama-cpp-python.")
    except Exception as e:
        logging.warning(f"Error during uninstall: {e}")
    
    logging.info("Building llama-cpp-python from source...")
    build_env = os.environ.copy()
    build_env.update(env_vars)
    if platform.system() == "Darwin" and platform.machine() == "arm64" and acc_type == "metal":
        build_env["CMAKE_ARGS"] = "-DGGML_METAL=on -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64"
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--verbose", "--force-reinstall",
        f"llama-cpp-python=={LATEST_VERSION}"
    ]
    try:
        result = run_subprocess(cmd, env=build_env, check=True)
        paths = get_data_paths()
        build_log = paths["logs"] / "build_output.log"
        with build_log.open("w", encoding="utf-8") as f:
            f.write("STDOUT:\n" + result["stdout"] + "\n\nSTDERR:\n" + result["stderr"])
        logging.info(f"Successfully built llama-cpp-python v{LATEST_VERSION}.")
        if acc_type in ["cuda", "metal"] and check_llama_cpp_cuda_support():
            logging.info("GPU acceleration support verified in the built package.")
        mark_step_complete("llama_cpp_install")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error building llama-cpp-python: {e}")
        logging.error(traceback.format_exc())
        return False

def install_torch_with_cuda() -> bool:
    """
    Install PyTorch with the appropriate backend.
    On macOS, install the default PyTorch build (which includes MPS support).
    On other platforms, install a CUDA-enabled build if available.
    """
    sys_platform = platform.system()
    if sys_platform == "Darwin":
        logging.info("=== Installing PyTorch for macOS ===")
        cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "torch", "torchvision", "torchaudio"]
        try:
            run_subprocess(cmd, check=True)
            verify = [
                sys.executable, "-c",
                "import torch; print('MPS available:', (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())); print('PyTorch version:', torch.__version__)"
            ]
            logging.info(run_subprocess(verify, check=False)["stdout"].strip())
            mark_step_complete("torch_install")
            return True
        except Exception as e:
            logging.error(f"Error installing PyTorch on macOS: {e}")
            return False
    else:
        logging.info("=== Installing PyTorch with CUDA support if available ===")
        cuda_version = "cu124"
        has_cuda = False
        try:
            proc = run_subprocess(["nvcc", "--version"], check=False)
            if proc["returncode"] == 0:
                has_cuda = True
                import re
                match = re.search(r"release (\d+\.\d+)", proc["stdout"])
                if match:
                    ver = float(match.group(1).split(".")[0])
                    cuda_version = "cu113" if ver < 11.0 else ("cu118" if ver < 12.0 else "cu124")
        except Exception as e:
            logging.warning(f"CUDA not detected: {e}")
        if has_cuda:
            cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir",
                   f"--index-url=https://download.pytorch.org/whl/{cuda_version}",
                   "torch", "torchvision", "torchaudio"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "-U", "--no-cache-dir", "torch", "torchvision", "torchaudio"]
        try:
            run_subprocess(cmd, check=True)
            logging.info("Successfully installed PyTorch with the appropriate backend.")
            mark_step_complete("torch_install")
            return True
        except Exception as e:
            logging.error(f"Error installing PyTorch: {e}")
            return False

def main():
    setup_logger()
    logging.info("KillerAI Agent setup utilities invoked directly.")
    logging.info("Acceleration configuration: " + str(get_acceleration_config()))
    install_llama_cpp_python()
    install_torch_with_cuda()
    logging.info("Setup complete.")