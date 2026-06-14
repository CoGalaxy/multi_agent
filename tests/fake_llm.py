from __future__ import annotations

import json

from verifiable_multi_agent.backends import LlmBackend


class FakeLlmBackend(LlmBackend):
    """Test-only backend with the same interface as a real LLM backend."""

    is_real_llm: bool = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, role: str | None = None) -> str:
        self.calls.append({"system": system, "user": user, "role": role})
        role_name = role or "agent"
        claim = f"{role_name} completed supported answer"
        return json.dumps(
            {
                "claim": claim,
                "evidence": [claim],
                "action": role_name,
                "uncertainty": 0.2,
            }
        )
