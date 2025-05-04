"""
LlamaCpp CLI implementation for KAIA.

This module provides a Python interface to the llama.cpp CLI executable.
"""

import os
import sys
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Tuple, Optional, Iterator, Callable

from killeraiagent.models.base import LLM, ModelInfo
from killeraiagent.paths import get_data_paths
from killeraiagent.hardware import AcceleratorType, detect_hardware_capabilities

logger = logging.getLogger(__name__)

class LlamaCppCLI(LLM):
    """
    Interface to llama.cpp CLI executable for inference.
    
    This class manages the subprocess for the llama-cli executable.
    """
    
    def __init__(
        self,
        model_info: ModelInfo,
        n_gpu_layers: int = 0,
        n_threads: int = 4,
        context_size: int = 4096,
        **kwargs
    ):
        """
        Initialize LlamaCpp CLI interface.
        
        Args:
            model_info: Model information
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            n_threads: Number of CPU threads to use
            context_size: Context window size in tokens
            **kwargs: Additional arguments
        """
        super().__init__(model_info, **kwargs)
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.context_size = context_size
        self.executable_path = None
        self.process = None
    
    def _find_executable(self) -> Optional[Path]:
        """
        Find the llama.cpp executable path.
        
        Returns:
            Path to executable or None if not found
        """
        paths = get_data_paths()
        bin_dir = paths.bin
        
        # Extensions based on platform
        if sys.platform == "win32":
            extensions = [".exe"]
        else:
            extensions = [""]
        
        # Try to find acceleration-specific executables first
        hw = detect_hardware_capabilities()
        acc_type = hw.primary_acceleration
        
        # Search for executables with the pattern: llama-cli[-ACCEL_TYPE][.exe]
        for ext in extensions:
            # Check for acceleration-specific binary
            if acc_type != AcceleratorType.CPU:
                exe_path = bin_dir / f"llama-cli-{acc_type.value}{ext}"
                if exe_path.exists():
                    logger.info(f"Found acceleration-specific executable: {exe_path}")
                    return exe_path
            
            # Check for generic binary
            exe_path = bin_dir / f"llama-cli{ext}"
            if exe_path.exists():
                logger.info(f"Found generic executable: {exe_path}")
                return exe_path
            
            # Check for alternative names
            for alt_name in ["main", "llama-main", "llama"]:
                exe_path = bin_dir / f"{alt_name}{ext}"
                if exe_path.exists():
                    logger.info(f"Found alternative executable: {exe_path}")
                    return exe_path
        
        logger.warning("No llama.cpp executable found")
        return None
    
    def load(self) -> bool:
        """
        Load the model. For CLI interface, this just checks if the executable exists.
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_loaded:
            return True
        
        # Check if model path exists
        if not self.model_info.model_path:
            logger.error("Model path not specified")
            return False
        
        model_path = Path(self.model_info.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Find executable
        self.executable_path = self._find_executable()
        if not self.executable_path:
            logger.error("llama.cpp executable not found. Please build or install it.")
            return False
        
        self.is_loaded = True
        return True
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> Tuple[str, Any]:
        """
        Generate text using llama.cpp CLI.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeated tokens
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        if not self.is_loaded:
            if not self.load():
                raise RuntimeError("Failed to load model")
        
        # Save prompt to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            prompt_file = f.name
            f.write(prompt)
        
        try:
            # Build command
            cmd = [
                str(self.executable_path),
                "-m", str(self.model_info.model_path),
                "-f", prompt_file,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "--top_p", str(top_p),
                "--repeat_penalty", str(repeat_penalty),
                "-t", str(self.n_threads),
                "-c", str(self.context_size),
                "--color", "0",  # Disable color output
                "-ngl", str(self.n_gpu_layers)
            ]
            
            # Add any additional arguments from kwargs
            for key, value in kwargs.items():
                if key == "top_k":
                    cmd.extend(["--top_k", str(value)])
                elif key == "frequency_penalty":
                    cmd.extend(["--frequency_penalty", str(value)])
                elif key == "presence_penalty":
                    cmd.extend(["--presence_penalty", str(value)])
                elif key == "mirostat":
                    cmd.extend(["--mirostat", str(value)])
                elif key == "mirostat_tau":
                    cmd.extend(["--mirostat_tau", str(value)])
                elif key == "mirostat_eta":
                    cmd.extend(["--mirostat_eta", str(value)])
                elif key == "seed":
                    cmd.extend(["--seed", str(value)])
            
            # Run process
            logger.info(f"Running llama.cpp: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            
            # Check for errors
            if result.returncode != 0:
                error_msg = f"llama.cpp process failed with code {result.returncode}: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Extract output (removing the prompt part)
            full_output = result.stdout.strip()
            
            # Skip the prompt part (everything before the first newline)
            output_lines = full_output.split("\n")
            if len(output_lines) > 1:
                # Skip lines until we're past the prompt
                prompt_completed = False
                result_lines = []
                
                for line in output_lines:
                    if not prompt_completed and line.strip() == "":
                        prompt_completed = True
                        continue
                    
                    if prompt_completed:
                        result_lines.append(line)
                
                # If we never found the end of the prompt, use the last 75% of the output
                if not prompt_completed:
                    output = full_output[int(len(full_output) * 0.25):]
                else:
                    output = "\n".join(result_lines)
            else:
                # If there's only one line, assume it's all output
                output = full_output
            
            # Metadata
            metadata = {
                "tokens": len(output.split()),
                "model": self.model_info.model_id,
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                    **kwargs
                }
            }
            
            return output, metadata
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(prompt_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary prompt file: {e}")
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        on_token: Optional[Callable] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream generated text using llama.cpp CLI.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeated tokens
            on_token: Callback function for each token
            **kwargs: Additional arguments
            
        Yields:
            Generated tokens
        """
        if not self.is_loaded:
            if not self.load():
                raise RuntimeError("Failed to load model")
        
        # Save prompt to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            prompt_file = f.name
            f.write(prompt)
        
        try:
            # Build command
            cmd = [
                str(self.executable_path),
                "-m", str(self.model_info.model_path),
                "-f", prompt_file,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "--top_p", str(top_p),
                "--repeat_penalty", str(repeat_penalty),
                "-t", str(self.n_threads),
                "-c", str(self.context_size),
                "--color", "0",  # Disable color output
                "-ngl", str(self.n_gpu_layers),
                "--streaming", "1"  # Enable streaming
            ]
            
            # Add any additional arguments from kwargs
            for key, value in kwargs.items():
                if key == "top_k":
                    cmd.extend(["--top_k", str(value)])
                elif key == "frequency_penalty":
                    cmd.extend(["--frequency_penalty", str(value)])
                elif key == "presence_penalty":
                    cmd.extend(["--presence_penalty", str(value)])
                elif key == "mirostat":
                    cmd.extend(["--mirostat", str(value)])
                elif key == "mirostat_tau":
                    cmd.extend(["--mirostat_tau", str(value)])
                elif key == "mirostat_eta":
                    cmd.extend(["--mirostat_eta", str(value)])
                elif key == "seed":
                    cmd.extend(["--seed", str(value)])
            
            # Run process
            logger.info(f"Running llama.cpp stream: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            
            # Store process for cleanup
            self.process = process
            
            # Skip the prompt part 
            prompt_completed = False
            prompt_len = len(prompt)
            chunk_buffer = ""
            
            # Read from stdout in real-time
            try:
                if process.stdout is None:
                    raise RuntimeError("Failed to start process or capture output stream")
                
                for line in process.stdout:
                    if not prompt_completed:
                        chunk_buffer += line
                        # Check if we've buffered more than the prompt length
                        if len(chunk_buffer) >= prompt_len:
                            # Skip past the prompt
                            excess = chunk_buffer[prompt_len:]
                            if excess:
                                if on_token:
                                    on_token(excess)
                                yield excess
                            prompt_completed = True
                            chunk_buffer = ""
                    else:
                        if on_token:
                            on_token(line)
                        yield line
            
            finally:
                # Make sure process is terminated
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                
                # Check for errors
                stderr = process.stderr.read() if process.stderr else ""
                if stderr and process.returncode != 0:
                    logger.error(f"llama.cpp process error: {stderr}")
                
                self.process = None
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(prompt_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary prompt file: {e}")
    
    def unload(self) -> None:
        """Unload the model and clean up resources."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        self.process = None
        self.is_loaded = False