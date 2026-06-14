"""LLM-driven agents used by the quant-only collaboration graph."""

from __future__ import annotations

import json
import logging

from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.contracts import AgentRole, ContractMessage

logger = logging.getLogger(__name__)


class LLMAgent:
    """A role-specific agent that asks a real LLM to produce ContractMessage JSON."""

    def __init__(self, role: AgentRole, backend: LlmBackend) -> None:
        if not backend.is_real_llm:
            raise RuntimeError("LLMAgent requires a real LLM backend.")
        self.role = role
        self.backend = backend

    def run(
        self,
        task: str,
        context: list[ContractMessage],
        subtask: str | None = None,
        action: str | None = None,
        stage_note: str | None = None,
    ) -> ContractMessage:
        system = self._build_system_prompt(task)
        user = self._build_user_prompt(task, context, subtask, stage_note)
        try:
            raw = self.backend.complete(system=system, user=user, role=self.role.value)
            data = json.loads(raw)
            return ContractMessage(
                role=self.role,
                subtask=subtask or "",
                claim=str(data.get("claim", "")),
                evidence=_ensure_str_list(data.get("evidence", [])),
                action=str(data.get("action", action or "")),
                uncertainty=float(data.get("uncertainty", 0.5)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("LLMAgent parse failed for role=%s: %s", self.role.value, exc)
            return ContractMessage(
                role=self.role,
                subtask=subtask or "",
                claim="parse error",
                evidence=[],
                action=action or "",
                uncertainty=1.0,
            )

    def _build_system_prompt(self, task: str) -> str:
        language = "Respond entirely in Simplified Chinese. " if _contains_cjk(task) else ""
        synthesis_rule = ""
        if self.role == AgentRole.SYNTHESIZER:
            synthesis_rule = (
                "When synthesizing the final answer, do not mention internal orchestration, "
                "topology, planner, executor, verifier, synthesizer, or trace unless the user asks. "
            )
        return (
            f"You are the {self.role.value} agent in a thesis prototype. "
            f"{language}"
            f"{synthesis_rule}"
            "Keep the original task as the only topic. Return only valid JSON."
        )

    @staticmethod
    def _build_user_prompt(
        task: str,
        context: list[ContractMessage],
        subtask: str | None,
        stage_note: str | None,
    ) -> str:
        context_summary = "\n".join(f"[{m.role.value}] {m.claim}" for m in context[-5:])
        parts = [
            f"Subtask: {subtask or 'process current graph node'}",
            "",
            f"Original task: {task}",
        ]
        if stage_note:
            parts.append(f"Stage note: {stage_note}")
        if context_summary:
            parts.append(f"Recent context:\n{context_summary}")
        parts.extend(
            [
                "",
                "Output format, strict JSON only:",
                "{",
                '  "claim": "one clear claim",',
                '  "evidence": ["evidence item 1", "evidence item 2"],',
                '  "action": "plan|execute|verify|synthesize",',
                '  "uncertainty": 0.0',
                "}",
            ]
        )
        return "\n".join(parts)


def _ensure_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)
