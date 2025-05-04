"""
Server-based implementation of llama.cpp model (LlamaCppServer).

Keeps the model loaded in memory as a server, for faster repeated requests.
"""

import os
import time
import logging
import subprocess
import requests
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from killeraiagent.models.base import LLM, ModelInfo
from killeraiagent.setup.setup_core import get_data_paths
from killeraiagent.hardware import detect_hardware_capabilities

logger = logging.getLogger(__name__)


class LlamaCppServer(LLM):
    """
    Implementation of llama.cpp as a persistent server process
    with an OpenAI-compatible API.
    """

    def __init__(
        self,
        model_info: ModelInfo,
        chat_format: str = "chatml",
        chat_template: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        seed: int = -1,
        grammar_file: Optional[str] = None,
        server_host: str = "127.0.0.1",
        server_port: int = 8080,
        n_parallel: int = 1,
        auto_start: bool = False
    ):
        """
        Args:
            model_info: Metadata about the model
            chat_format: Chat format style (e.g. 'chatml', 'llama2', etc.)
            chat_template: Jinja2 template for chat
            n_ctx: Context length
            n_threads: CPU threads
            n_gpu_layers: GPU-accelerated layers
            seed: Random seed
            grammar_file: Optional grammar file path
            server_host: Host for the local server
            server_port: Port for the local server
            n_parallel: How many requests to handle in parallel
            auto_start: If True, start the server in constructor
        """
        super().__init__(model_info)
        self.chat_format = chat_format
        self.chat_template = chat_template

        self.n_ctx = n_ctx if n_ctx is not None else model_info.context_length
        self.n_threads = n_threads if n_threads is not None else (os.cpu_count() or 4)
        self.n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None
            else (-1 if model_info.requires_gpu else 0)
        )
        self.seed = seed
        self.grammar_file = grammar_file

        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        self.n_parallel = n_parallel

        self.paths = get_data_paths()
        self.bin_dir = self.paths.bin
        self.temp_dir = self.paths.temp

        # Path to the llama server executable
        self.llama_server_path = self.bin_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if not self.llama_server_path.exists():
            self.llama_server_path = self.bin_dir / ("server.exe" if os.name == "nt" else "server")

        # Persistent process
        self.server_process: Optional[subprocess.Popen] = None
        self.health_check_thread: Optional[threading.Thread] = None
        self.health_check_stop = threading.Event()

        # Adjust GPU layers
        self._adjust_gpu_layers()

        if auto_start:
            ok = self.load()
            if not ok:
                logger.error("Failed to auto-start the llama.cpp server.")

    def _adjust_gpu_layers(self) -> None:
        hw_caps = detect_hardware_capabilities()
        acc_type = hw_caps.primary_acceleration.value
        if acc_type == "cpu" and self.n_gpu_layers != 0:
            self.n_gpu_layers = 0
            logger.info("CPU-only acceleration, set n_gpu_layers=0")
        if acc_type == "metal" and self.n_gpu_layers > 0:
            logger.info(f"Metal acceleration with {self.n_gpu_layers} GPU layers")

    def load(self) -> bool:
        """
        Start the server if not already running.
        Returns True if successfully started/connected, else False.
        """
        if self.is_loaded and self.server_process and self.server_process.poll() is None:
            return True

        # Check model file and server exe
        if not self.model_info.model_path or not Path(self.model_info.model_path).exists():
            logger.error(f"Model file not found: {self.model_info.model_path}")
            return False
        if not self.llama_server_path.exists():
            logger.error(f"Server executable not found: {self.llama_server_path}")
            return False

        return self._start_server()

    def _start_server(self) -> bool:
        if self.server_process is not None:
            logger.warning("Server already running, stopping first.")
            self.close()

        cmd = self._build_server_command()
        logger.info(f"Starting llama.cpp server: {' '.join(cmd)}")

        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            logger.error(f"Failed to start server process: {e}")
            self.server_process = None
            return False

        # Wait up to 30s for /v1/models to return 200
        start_time = time.time()
        while time.time() - start_time < 30:
            if self.server_process.poll() is not None:
                err_out = ""
                if self.server_process.stderr:
                    err_out = self.server_process.stderr.read()
                logger.error(f"Server exited early with code {self.server_process.returncode}\n{err_out}")
                return False

            try:
                resp = requests.get(f"{self.server_url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    logger.info("Server started successfully.")
                    self.is_loaded = True
                    # Start health check thread
                    self._start_health_check_thread()
                    return True
            except requests.RequestException:
                pass

            time.sleep(0.5)

        logger.error("Timed out waiting for server to respond.")
        return False

    def _start_health_check_thread(self) -> None:
        if self.health_check_thread and self.health_check_thread.is_alive():
            return
        self.health_check_stop.clear()
        self.health_check_thread = threading.Thread(
            target=self._health_check_worker,
            daemon=True
        )
        self.health_check_thread.start()

    def _health_check_worker(self, interval: int = 30) -> None:
        while not self.health_check_stop.is_set():
            if self.server_process and self.server_process.poll() is not None:
                code = self.server_process.returncode
                logger.warning(f"Server died (code {code}); restarting...")
                self._start_server()
            else:
                try:
                    resp = requests.get(f"{self.server_url}/v1/models", timeout=2)
                    if resp.status_code != 200:
                        logger.warning(f"Health check got status {resp.status_code}, restarting.")
                        self._start_server()
                except requests.RequestException:
                    logger.warning("Health check request failed, restarting server.")
                    self._start_server()
            self.health_check_stop.wait(interval)

    def close(self) -> None:
        """
        Stop the server and free resources.
        """
        if self.health_check_thread:
            self.health_check_stop.set()
            self.health_check_thread.join(timeout=2)
        if self.server_process is not None:
            logger.info("Stopping llama.cpp server.")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, forcing kill.")
                self.server_process.kill()
            self.server_process = None
        self.is_loaded = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1
    ) -> Tuple[str, Any]:
        """
        Call the local server /v1/completions (for a standard prompt).
        """
        if not self.load():
            return ("Error: LlamaCppServer not running", None)

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty - 1.0,
            "stream": False
        }

        try:
            resp = requests.post(f"{self.server_url}/v1/completions", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices"):
                return (data["choices"][0].get("text", ""), data)
            return ("", data)
        except Exception as e:
            logger.error(f"Error calling server: {e}")
            return (f"Error: {e}", {"error": str(e)})

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1
    ) -> Dict[str, Any]:
        """
        Call /v1/chat/completions with a list of role/content messages.
        """
        if not self.load():
            return {"error": "Server not running, cannot generate chat completion"}

        payload = {
            "model": self.model_info.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty - 1.0,
            "stream": False
        }

        try:
            resp = requests.post(f"{self.server_url}/v1/chat/completions", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Error in generate_chat_completion: {e}")
            return {"error": str(e)}

    def _build_server_command(self) -> List[str]:
        cmd = [
            str(self.llama_server_path),
            "-m", str(self.model_info.model_path),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-ngl", str(self.n_gpu_layers),
            "--host", self.server_host,
            "--port", str(self.server_port),
        ]

        if self.n_parallel > 1:
            cmd.extend(["-np", str(self.n_parallel)])
        if self.chat_format:
            cmd.extend(["--chat-format", self.chat_format])
        if self.chat_template:
            template_file = self.temp_dir / f"server_template_{int(time.time())}.txt"
            with open(template_file, 'w') as f:
                f.write(self.chat_template)
            cmd.extend(["--chat-template", str(template_file)])
        if self.grammar_file:
            cmd.extend(["--grammar-file", str(self.grammar_file)])

        return cmd
    
    def unload(self) -> None:
        """Implementation of abstract unload method. Nothing to unload for API-based models."""
        self.is_loaded = False