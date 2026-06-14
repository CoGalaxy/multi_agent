"""Contract verification for the thesis prototype.

Layer 1 is a small rule-based verifier. It checks structural completeness,
keyword support, explicit verifier rejection, task coverage, and soft
claim/evidence contradiction hints.

Layer 2 is an optional LLM semantic verifier kept for experiments.
"""

from __future__ import annotations

import json
import logging
import re

from verifiable_multi_agent.contracts import AgentRole, ContractMessage, VerificationResult

logger = logging.getLogger(__name__)

_SEMANTIC_PROMPT = """You are a contract verifier. Audit this multi-agent trace.

Original task: {task}

Collaboration trace:
{trace}

Check for CRITICAL issues only (do not flag minor style problems):

1. TOPIC DRIFT: Does any agent solve a DIFFERENT problem than requested?
   - Only flag if the agent clearly addresses the wrong task.
   - Do NOT flag minor imprecision in wording.
2. LOGICAL GAP: Does the executor skip a required step or do something contradictory?
   - Flag if a step from the planner is completely missing in execution.
3. MISSING VERIFICATION: If the task requires checking/verifying something, was it actually verified?
   - Flag if the task says "verify X" but no one actually checks X.
4. INCOMPLETE: Did the trace fail to address any part of the original task?

Do NOT reject for:
- Generic-sounding evidence unless it causes a real problem
- Minor wording issues that do not affect correctness
- The trace being short

Output ONLY a valid JSON object (no markdown, no explanation):

{{
  "accepted": true or false,
  "score": 0.0 to 1.0,
  "violations": ["specific issue if any, empty list if clean"],
  "next_action": "accept" or "escalate"
}}

JSON:"""

_POSITIVE_CLAIM_KEYWORDS = ("correct", "valid", "passed", "\u6b63\u786e", "\u65e0\u8bef", "\u901a\u8fc7")
_NEGATIVE_EVIDENCE_KEYWORDS = ("error", "fail", "failed", "failure", "\u9519\u8bef", "\u5931\u8d25")


def verify_contracts(messages: list[ContractMessage], task: str | None = None) -> VerificationResult:
    return ContractVerifier().verify(messages, task=task)


class ContractVerifier:
    """Rule-based verifier used by the runtime contract gate."""

    def verify(self, messages: list[ContractMessage], task: str | None = None) -> VerificationResult:
        if not messages:
            return VerificationResult(
                accepted=False,
                support_rate=0.0,
                violations=["trace_empty"],
                next_action="retry",
            )

        violations: list[str] = []
        total_evidence_items = 0
        supported_evidence_items = 0

        for message in messages:
            total_evidence_items += len(message.evidence)
            for evidence in message.evidence:
                if _evidence_supports_claim(evidence, message.claim):
                    supported_evidence_items += 1

            if message.role == AgentRole.VERIFIER and "REJECTED" in message.claim.upper():
                violations.append(f"verifier_rejected:{message.id}:{message.claim}")

            if _has_claim_evidence_mismatch(message):
                violations.append(f"soft_claim_evidence_mismatch:{message.role.value}:{message.id}")

            if not message.evidence:
                violations.append(f"{message.role.value}:{message.id}:missing_evidence")
            if not message.claim.strip():
                violations.append(f"{message.role.value}:{message.id}:empty_claim")
            if not message.action.strip():
                violations.append(f"{message.role.value}:{message.id}:missing_action")
            if message.uncertainty > 0.8 and message.budget_hint <= 0:
                violations.append(f"{message.role.value}:{message.id}:high_uncertainty_without_budget")

        support_rate = (
            supported_evidence_items / total_evidence_items
            if total_evidence_items > 0
            else 0.0
        )
        task_coverage, coverage_violations = _task_coverage(task, messages)
        violations.extend(coverage_violations)
        semantic_penalties = sum(
            1
            for item in violations
            if item.startswith("verifier_rejected")
        )
        if semantic_penalties:
            support_rate = max(0.0, support_rate - 0.25 * semantic_penalties)
        soft_mismatches = sum(1 for item in violations if item.startswith("soft_claim_evidence_mismatch"))
        if soft_mismatches:
            support_rate = max(0.0, support_rate - 0.05 * soft_mismatches)

        accepted = (
            support_rate >= 0.5
            and (task_coverage is None or task_coverage >= 0.75)
            and not any("missing_action" in item for item in violations)
            and not any(item.startswith("verifier_rejected") for item in violations)
            and not any(item.startswith("missing_task_coverage") for item in violations)
        )
        return VerificationResult(
            accepted=accepted,
            support_rate=round(support_rate, 3),
            task_coverage=task_coverage,
            violations=violations,
            next_action="accept" if accepted else "escalate",
        )


def _has_claim_evidence_mismatch(message: ContractMessage) -> bool:
    claim_lower = message.claim.lower()
    if not any(keyword in claim_lower for keyword in _POSITIVE_CLAIM_KEYWORDS):
        return False
    for evidence in message.evidence:
        evidence_lower = evidence.lower()
        if any(keyword in evidence_lower for keyword in _NEGATIVE_EVIDENCE_KEYWORDS):
            return True
    return False


def _task_coverage(task: str | None, messages: list[ContractMessage]) -> tuple[float | None, list[str]]:
    if not task or not task.strip():
        return None, []
    requirements = _infer_task_requirements(task)
    if not requirements:
        return None, []

    answer_text = _answer_text(messages)
    satisfied = 0
    violations: list[str] = []
    for requirement, keywords in requirements:
        if _contains_any(answer_text, keywords):
            satisfied += 1
        else:
            violations.append(f"missing_task_coverage:{requirement}")
    return round(satisfied / len(requirements), 3), violations


def _infer_task_requirements(task: str) -> list[tuple[str, tuple[str, ...]]]:
    text = task.lower()
    requirements: list[tuple[str, tuple[str, ...]]] = []
    rules: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        (
            "design",
            ("design", "schema", "architecture", "\u8bbe\u8ba1", "\u67b6\u6784", "\u8868\u7ed3\u6784"),
            ("schema", "table", "field", "constraint", "ddl", "create table", "\u8868", "\u5b57\u6bb5", "\u7ea6\u675f"),
        ),
        (
            "code",
            (
                "write code", "implement", "function", "interface", "api", "code",
                "\u5199\u4ee3\u7801", "\u7f16\u5199\u4ee3\u7801", "\u5b9e\u73b0", "\u51fd\u6570", "\u63a5\u53e3", "\u4ee3\u7801",
            ),
            ("```", "def ", "class ", "return ", "post ", "get ", "@app", "request.", "response.", "async def "),
        ),
        (
            "test",
            (
                "test", "unit test", "pytest", "validate test",
                "\u6d4b\u8bd5", "\u5355\u5143\u6d4b\u8bd5", "\u6d4b\u8bd5\u7528\u4f8b",
            ),
            ("test_", "assert ", "pytest", "unittest", "assert(", "self.assert"),
        ),
        (
            "compare",
            ("compare", "versus", "vs", "\u6bd4\u8f83", "\u5bf9\u6bd4", "\u5dee\u5f02"),
            ("difference", "compare", "dimension", "versus", "\u5dee\u5f02", "\u5bf9\u6bd4", "\u7ef4\u5ea6"),
        ),
        (
            "explain",
            ("explain", "why", "\u89e3\u91ca", "\u8bf4\u660e", "\u4e3a\u4ec0\u4e48"),
            ("because", "means", "\u56e0\u4e3a", "\u8868\u793a", "\u610f\u5473", "\u8bf4\u660e"),
        ),
        (
            "proof",
            ("prove", "proof", "derive", "derivation", "\u8bc1\u660e", "\u63a8\u5bfc"),
            ("assume", "therefore", "qed", "step", "\u5047\u8bbe", "\u56e0\u4e3a", "\u6240\u4ee5", "\u5f97\u8bc1", "\u6b65\u9aa4"),
        ),
        (
            "summarize",
            ("summarize", "summary", "\u603b\u7ed3", "\u6982\u62ec"),
            ("summary", "conclusion", "\u603b\u7ed3", "\u7ed3\u8bba", "\u6982\u62ec"),
        ),
        (
            "query",
            ("query", "lookup", "search", "\u67e5\u8be2", "\u68c0\u7d22", "\u641c\u7d22"),
            ("query", "result", "where", "select", "\u67e5\u8be2", "\u7ed3\u679c", "\u6761\u4ef6"),
        ),
        (
            "safety",
            ("security", "risk", "permission", "\u5b89\u5168", "\u98ce\u9669", "\u6743\u9650"),
            ("risk", "permission", "safe", "\u98ce\u9669", "\u6743\u9650", "\u5b89\u5168"),
        ),
    ]
    for name, triggers, expected in rules:
        if _contains_any(text, triggers):
            requirements.append((name, expected))
    return _dedupe_requirements(requirements)


def _answer_text(messages: list[ContractMessage]) -> str:
    synthesizer_messages = [
        message
        for message in messages
        if message.role == AgentRole.SYNTHESIZER or message.metadata.get("node_type") == "synthesizer"
    ]
    selected = synthesizer_messages[-1:] if synthesizer_messages else messages[-2:]
    parts: list[str] = []
    for message in selected:
        parts.append(message.claim)
        parts.extend(message.evidence)
        parts.append(message.action)
    return "\n".join(parts).lower()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _dedupe_requirements(requirements: list[tuple[str, tuple[str, ...]]]) -> list[tuple[str, tuple[str, ...]]]:
    seen: set[str] = set()
    deduped: list[tuple[str, tuple[str, ...]]] = []
    for name, keywords in requirements:
        if name not in seen:
            seen.add(name)
            deduped.append((name, keywords))
    return deduped


def _evidence_supports_claim(evidence: str, claim: str) -> bool:
    if not evidence.strip() or not claim.strip():
        return False
    ev_lower = evidence.lower()
    claim_lower = claim.lower()

    ev_en = set(re.findall(r"[a-zA-Z_]{3,}", ev_lower))
    claim_en = set(re.findall(r"[a-zA-Z_]{3,}", claim_lower))
    if len(ev_en & claim_en) >= 2:
        return True

    matched_bigrams: set[str] = set()
    for index in range(len(evidence) - 1):
        bigram = evidence[index:index + 2]
        if _is_cjk_bigram(bigram) and bigram in claim:
            matched_bigrams.add(bigram)
            if len(matched_bigrams) >= 2:
                return True
    return False


def _is_cjk_bigram(value: str) -> bool:
    return len(value) == 2 and all("\u4e00" <= char <= "\u9fff" for char in value)


class SemanticVerifier:
    """Optional LLM semantic verifier for thesis experiments."""

    def __init__(self, backend) -> None:
        self._backend = backend

    def verify(self, task: str, messages: list[ContractMessage]) -> VerificationResult:
        if not messages:
            return VerificationResult(
                accepted=False,
                support_rate=0.0,
                violations=["trace_empty"],
                next_action="retry",
            )

        trace_lines = []
        for index, message in enumerate(messages):
            trace_lines.append(
                f"[{index}] role={message.role.value} "
                f"claim={json.dumps(message.claim, ensure_ascii=False)} "
                f"evidence={json.dumps(message.evidence, ensure_ascii=False)} "
                f"action={json.dumps(message.action, ensure_ascii=False)} "
                f"uncertainty={message.uncertainty}"
            )
        trace_text = "\n".join(trace_lines)

        prompt = _SEMANTIC_PROMPT.format(task=task, trace=trace_text)
        try:
            raw = self._backend.complete(system="", user=prompt)
            data = json.loads(raw)
            return VerificationResult(
                accepted=bool(data.get("accepted", False)),
                support_rate=_compute_support_rate(messages),
                violations=data.get("violations", []),
                next_action=data.get("next_action", "accept"),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Semantic verifier JSON parse failed: %s", exc)
            return VerificationResult(
                accepted=True,
                support_rate=_compute_support_rate(messages),
                violations=[f"semantic_verifier_parse_error: {exc}"],
                next_action="accept",
            )


def _compute_support_rate(messages: list[ContractMessage]) -> float:
    total = 0
    supported = 0
    for message in messages:
        total += len(message.evidence)
        for evidence in message.evidence:
            if _evidence_supports_claim(evidence, message.claim):
                supported += 1
    return round(supported / total, 3) if total > 0 else 0.0
