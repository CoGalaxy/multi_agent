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
            data = _loads_contract_json(raw)
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
        verifier_rule = ""
        if self.role == AgentRole.VERIFIER:
            verifier_rule = (
                'For verifier output, the claim field must start with "APPROVED" or '
                '"REJECTED: [reason]". If rejecting, name the specific upstream claim '
                "and the specific evidence item that contradicts it. "
            )
        if self.role == AgentRole.SYNTHESIZER:
            synthesis_rule = (
                "When synthesizing the final answer, do not mention internal orchestration, "
                "topology, planner, executor, verifier, synthesizer, or trace unless the user asks. "
                "Produce a substantive final answer, not a one-sentence summary. "
                "Use the upstream evidence to give clear dimensions, comparisons, caveats, and a conclusion. "
                "If upstream nodes produced concrete deliverables, the final answer must include the complete "
                "deliverables instead of replacing them with a description. Concrete deliverables include schema, "
                "SQL, API code, functions, tests, configuration, step-by-step procedures, tables, and checklists. "
            )
        return (
            f"You are the {self.role.value} agent in a thesis prototype. "
            f"{language}"
            f"{verifier_rule}"
            f"{synthesis_rule}"
            "Keep the original task as the only topic. Return only valid JSON."
        )

    def _build_user_prompt(
        self,
        task: str,
        context: list[ContractMessage],
        subtask: str | None,
        stage_note: str | None,
    ) -> str:
        context_summary = "\n".join(_format_context_message(message) for message in context[-6:])
        parts = [
            f"Subtask: {subtask or 'process current graph node'}",
            "",
            f"Original task: {task}",
        ]
        if stage_note:
            parts.append(f"Stage note: {stage_note}")
        if context_summary:
            parts.append(f"Recent context:\n{context_summary}")
        deliverable_instruction = _deliverable_instruction(task, subtask)
        if deliverable_instruction:
            parts.append(f"Deliverable requirement: {deliverable_instruction}")
        parts.extend(
            [
                "",
                "Output format, strict JSON only:",
                "{",
                _claim_schema_line(self.role, task, subtask),
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


def _loads_contract_json(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(_escape_control_chars_inside_strings(raw))
    if not isinstance(data, dict):
        raise TypeError("LLM response JSON must be an object")
    return data


def _escape_control_chars_inside_strings(raw: str) -> str:
    """Repair common LLM JSON mistakes such as literal newlines in string values."""
    repaired: list[str] = []
    in_string = False
    escaped = False
    for char in raw:
        if escaped:
            repaired.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            repaired.append(char)
            escaped = True
            continue
        if char == '"':
            repaired.append(char)
            in_string = not in_string
            continue
        if in_string and char == "\n":
            repaired.append("\\n")
            continue
        if in_string and char == "\r":
            repaired.append("\\r")
            continue
        if in_string and char == "\t":
            repaired.append("\\t")
            continue
        repaired.append(char)
    return "".join(repaired)


def _format_context_message(message: ContractMessage) -> str:
    evidence = "; ".join(message.evidence[:3])
    if evidence:
        return f"[{message.role.value}] claim={message.claim}\n  evidence={evidence}"
    return f"[{message.role.value}] claim={message.claim}"


def _deliverable_instruction(task: str, subtask: str | None) -> str:
    text = f"{task} {subtask or ''}".lower()
    instructions: list[str] = []
    if _contains_any_word(text, ("prove", "proof", "derive", "derivation", "\u8bc1\u660e", "\u63a8\u5bfc")):
        instructions.append(
            "Return the complete proof or derivation with assumptions, steps, and conclusion; do not only describe that a proof was provided."
        )
    if _contains_any_word(text, ("implement", "write code", "function", "api", "\u5b9e\u73b0", "\u5199", "\u51fd\u6570", "\u63a5\u53e3")):
        instructions.append(
            "Return the complete implementation as fenced code blocks; do not replace code with prose."
        )
    if _contains_any_word(text, ("schema", "database", "table", "\u6570\u636e\u5e93", "\u8868", "\u8bbe\u8ba1")):
        instructions.append(
            "Return the complete schema, preferably as SQL DDL or an explicit table/field/constraint list."
        )
    if _contains_any_word(text, ("test", "pytest", "unit test", "\u6d4b\u8bd5", "\u7528\u4f8b", "\u9a8c\u8bc1")):
        instructions.append(
            "Return concrete test cases with assertions; do not only say that tests were written."
        )
    if _contains_any_word(text, ("list", "table", "checklist", "plan", "\u5217\u51fa", "\u8868\u683c", "\u6e05\u5355", "\u8ba1\u5212")):
        instructions.append(
            "Return the requested table, list, checklist, or plan as the actual deliverable, not a summary that it exists."
        )
    return " ".join(instructions)


def _claim_schema_line(role: AgentRole, task: str = "", subtask: str | None = None) -> str:
    deliverable_instruction = _deliverable_instruction(task, subtask)
    if role == AgentRole.VERIFIER:
        return (
            '  "claim": "APPROVED if all checked claims are supported; or '
            'REJECTED: [reason] with the exact conflicting claim and evidence item",'
        )
    if role == AgentRole.SYNTHESIZER:
        return (
            '  "claim": "complete final answer in Simplified Chinese when the task is Chinese; '
            'write 3-6 compact paragraphs or bullet points; include key dimensions, supported details, '
            'and a direct conclusion; preserve complete upstream deliverables such as schema, SQL, API code, '
            'functions, tests, configuration, procedures, tables, or checklists",'
        )
    if deliverable_instruction:
        return (
            '  "claim": "complete concrete deliverable requested by the task, such as proof steps, derivation, '
            'schema, SQL, API implementation, functions, tests, tables, checklists, or plans; '
            'do not replace deliverables with prose about having done them",'
        )
    return '  "claim": "one clear claim",'


def _contains_any_word(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)
