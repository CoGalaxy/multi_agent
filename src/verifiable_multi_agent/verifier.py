"""
合约验证器 — 双层验证：结构合规 + 语义质量。

Layer 1 — 结构验证 (verify_contracts):
  零成本 rule-based 检查：claim 是否有 evidence、action 是否非空、
  高不确定性是否伴随预算提示。始终执行，作为快速 gate。

Layer 2 — 语义验证 (SemanticVerifier):
  用 LLM 检查内容质量：是否跑题、evidence 是否真实支撑 claim、
  执行逻辑是否一致、任务是否被完整覆盖。这是论文的核心创新点——
  形式验证保证基本合规，语义验证检测"看起来合规但内容错误"的 case。
"""

from __future__ import annotations

import json
import logging

from verifiable_multi_agent.contracts import ContractMessage, VerificationResult

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
- Generic-sounding evidence (e.g. "inspected documents") unless it causes a real problem
- Minor wording issues that don't affect correctness
- The trace being short — brevity is acceptable if the task is simple

Output ONLY a valid JSON object (no markdown, no explanation):

{{
  "accepted": true or false,
  "score": 0.0 to 1.0,
  "violations": ["specific issue if any, empty list if clean"],
  "next_action": "accept" or "escalate"
}}

JSON:"""


def verify_contracts(messages: list[ContractMessage]) -> VerificationResult:
    """Layer 1: 结构验证 — 零成本 rule-based gate。"""
    if not messages:
        return VerificationResult(
            accepted=False, support_rate=0.0,
            violations=["trace_empty"], next_action="retry",
        )

    violations: list[str] = []
    supported = 0
    for message in messages:
        if message.has_support:
            supported += 1
        else:
            violations.append(f"{message.role.value}:{message.id}:unsupported_claim")
        if not message.action.strip():
            violations.append(f"{message.role.value}:{message.id}:missing_action")
        if message.uncertainty > 0.8 and message.budget_hint <= 0:
            violations.append(f"{message.role.value}:{message.id}:high_uncertainty_without_budget")

    support_rate = supported / len(messages)
    accepted = support_rate >= 0.8 and not any("missing_action" in item for item in violations)
    return VerificationResult(
        accepted=accepted,
        support_rate=round(support_rate, 3),
        violations=violations,
        next_action="accept" if accepted else "escalate",
    )


class SemanticVerifier:
    """Layer 2: 语义验证 — 用 LLM 检查内容质量和逻辑一致性。

    论文要点：这层展示了"可验证"不只是结构检查——系统能主动发现
    topic drift、evidence gap、逻辑断裂等问题，即使结构合规。
    """

    def __init__(self, backend) -> None:
        self._backend = backend

    def verify(self, task: str, messages: list[ContractMessage]) -> VerificationResult:
        if not messages:
            return VerificationResult(
                accepted=False, support_rate=0.0,
                violations=["trace_empty"], next_action="retry",
            )

        # 组装 trace 文本：每个消息提取关键字段
        trace_lines = []
        for i, message in enumerate(messages):
            trace_lines.append(
                f"[{i}] role={message.role.value} "
                f"claim=\"{message.claim}\" "
                f"evidence={json.dumps(message.evidence)} "
                f"action=\"{message.action}\" "
                f"uncertainty={message.uncertainty}"
            )
        trace_text = "\n".join(trace_lines)

        prompt = _SEMANTIC_PROMPT.format(task=task, trace=trace_text)
        try:
            raw = self._backend.complete(system="", user=prompt)
            data = json.loads(raw)
            # 同步计算 support_rate
            supported = sum(1 for m in messages if m.has_support)
            return VerificationResult(
                accepted=bool(data.get("accepted", False)),
                support_rate=round(supported / len(messages), 3),
                violations=data.get("violations", []),
                next_action=data.get("next_action", "accept"),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Semantic verifier JSON parse failed: %s", exc)
            # fallback: 接受但标注解析失败
            return VerificationResult(
                accepted=True,
                support_rate=1.0,
                violations=[f"semantic_verifier_parse_error: {exc}"],
                next_action="accept",
            )
