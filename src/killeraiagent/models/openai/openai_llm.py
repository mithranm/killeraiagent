"""
OpenAI-based LLM class (OpenAILLM).

Uses the remote OpenAI API for completions and chat, no local resources to manage.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import requests

from killeraiagent.models.base import LLM, ModelInfo

logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    """
    LLM for the OpenAI remote API (chat/completions).
    """

    def __init__(
        self,
        model_info: ModelInfo,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        default_model: Optional[str] = None,
        request_timeout: int = 120
    ):
        """
        Args:
            model_info: Basic info about the model (though we mostly use default_model).
            api_key: The OpenAI API key (Bearer token).
            base_url: The base URL for the API (can override for dev or proxies).
            default_model: e.g. "gpt-3.5-turbo"
            request_timeout: Request timeout in seconds.
        """
        super().__init__(model_info)
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model or model_info.model_id
        self.request_timeout = request_timeout

    def load(self) -> bool:
        """
        No local resources. We just mark it loaded if we have an API key.
        """
        if self.is_loaded:
            return True
        if not self.api_key:
            logger.warning("OpenAI API key not provided.")
        self.is_loaded = True
        return True

    def close(self) -> None:
        """
        No local resources to free. Just reset is_loaded if you want.
        """
        self.is_loaded = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0
    ) -> Tuple[str, Any]:
        """
        Create a simple text completion via /v1/completions.
        """
        if not self.load():
            return ("Error: No API key or not loaded", None)

        url = f"{self.base_url}/completions"
        headers = self._get_headers()
        payload = {
            "model": self.default_model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            # repeat_penalty is not exactly in OpenAI API,
            # you might approximate via frequency_penalty or presence_penalty
            "frequency_penalty": repeat_penalty - 1.0 if repeat_penalty != 1.0 else 0.0
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices"):
                return (data["choices"][0].get("text", ""), data)
            return ("", data)
        except Exception as e:
            logger.error(f"OpenAI generate error: {e}")
            return (f"Error: {e}", {"error": str(e)})

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create a chat completion via /v1/chat/completions.
        """
        if not self.load():
            return {"error": "OpenAILLM not loaded or no API key"}

        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        payload = {
            "model": self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty - 1.0 if repeat_penalty != 1.0 else 0.0
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"OpenAI chat completion error: {e}")
            return {"error": str(e)}

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def unload(self) -> None:
        """Implementation of abstract unload method. Nothing to unload for API-based models."""
        self.is_loaded = False