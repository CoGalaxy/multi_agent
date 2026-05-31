"""
LLM 后端抽象层 — 解耦 Agent 逻辑与模型调用。

设计意图：
- Mock/Echo 模式：零依赖跑通管线，中期前验证用
- OpenAI-compatible 模式：对接 vLLM/local-model 等本地推理服务
- DeepSeek 模式：过渡期使用云端 API，最终替换为本地模型

学术论文的最终配置：用 7B 量级模型通过 vLLM 自托管，
所有 API 调用走 http://127.0.0.1:8000/v1 —— 无外部依赖，
可复现，成本可控。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any

import httpx


class LlmBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, role: str | None = None) -> str:
        raise NotImplementedError


class EchoBackend(LlmBackend):
    """调试用后端：直接拼接 system + user 返回，不调用任何模型。"""

    def complete(self, system: str, user: str, role: str | None = None) -> str:
        return f"{system.strip()} :: {user.strip()}"


class OpenAICompatibleBackend(LlmBackend):
    """
    通用 OpenAI-compatible 后端。

    支持 vLLM、Ollama、text-generation-webui 等所有兼容 /v1/chat/completions
    的本地推理服务。接入本地模型后这是主要使用的后端。

    role_models 机制允许不同角色使用不同模型（如 verifier 用强模型），
    这是成本-质量权衡的关键设计点。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "local-model",
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        role_models: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        max_retries: int = 2,
        retry_sleep: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.role_models = role_models or {}
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def complete(self, system: str, user: str, role: str | None = None) -> str:
        selected_model = self.role_models.get(role or "", self.model)
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        payload.update(self.extra_body)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        # 指数退避重试：处理本地模型服务的偶发连接波动
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(
            f"LLM backend request failed after {self.max_retries + 1} attempts "
            f"(model={selected_model}, role={role or 'default'}): {last_error}"
        ) from last_error


class DeepSeekBackend(OpenAICompatibleBackend):
    """
    DeepSeek 云端后端 — 中期开发阶段的过渡方案。

    最终论文中应替换为本地模型。保留此类用于：
    1. 与本地小模型做性能对比实验
    2. 在本地模型不可用时提供 fallback

    judge_model 机制：verifier 和 synthesizer 需要更强的推理能力，
    可以用不同规格的模型（如 flash vs pro），这是成本-质量权衡的
    一个实验变量。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        judge_model: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 120.0,
        max_retries: int = 2,
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
            max_retries=max_retries,
        )


class OllamaBackend(LlmBackend):
    """
    Ollama 本地后端 — 最便捷的本地推理方案。

    使用 Ollama 原生 /api/chat 端点（非 OpenAI-compatible），以支持
    think=False 关闭 reasoning 模型的思考模式。

    Qwen3.5 默认会消耗大量 token 在 thinking 上；设置 think=False 后
    直接输出回答，大幅降低延迟和 token 消耗。
    """

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        think: bool = False,
        max_retries: int = 2,
        retry_sleep: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = think
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def complete(self, system: str, user: str, role: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        # 关闭 reasoning 模型的 thinking 模式
        if not self.think:
            payload["think"] = False

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(
            f"Ollama request failed after {self.max_retries + 1} attempts "
            f"(model={self.model}, role={role or 'default'}): {last_error}"
        ) from last_error
