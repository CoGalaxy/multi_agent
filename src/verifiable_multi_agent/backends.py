from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx


class LlmBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, role: str | None = None) -> str:
        raise NotImplementedError


class EchoBackend(LlmBackend):
    def complete(self, system: str, user: str, role: str | None = None) -> str:
        return f"{system.strip()} :: {user.strip()}"


class OpenAICompatibleBackend(LlmBackend):
    """Backend for OpenAI-compatible `/v1/chat/completions` servers."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "local-model",
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        role_models: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.role_models = role_models or {}
        self.extra_body = extra_body or {}

    def complete(self, system: str, user: str, role: str | None = None) -> str:
        payload = {
            "model": self.role_models.get(role or "", self.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        payload.update(self.extra_body)
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


class DeepSeekBackend(OpenAICompatibleBackend):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        judge_model: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 120.0,
    ) -> None:
        role_models = {}
        if judge_model:
            role_models = {
                "verifier": judge_model,
                "synthesizer": judge_model,
            }
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            role_models=role_models,
        )
