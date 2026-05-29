from __future__ import annotations

from abc import ABC, abstractmethod

import httpx


class LlmBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class EchoBackend(LlmBackend):
    def complete(self, system: str, user: str) -> str:
        return f"{system.strip()} :: {user.strip()}"


class OpenAICompatibleBackend(LlmBackend):
    """Backend for vLLM's OpenAI-compatible `/v1/chat/completions` server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "local-model",
        api_key: str = "EMPTY",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
