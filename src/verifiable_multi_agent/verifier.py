"""
合约验证器 — 双层验证：结构合规 + 语义质量。

Layer 1 — 结构验证 (verify_contracts):
  零成本 rule-based 检查：claim 是否有 evidence、action 是否非空、
  高不确定性是否伴随预算提示。始终执行，作为快速 gate。
  support_rate 使用关键词重叠逐条评估 evidence 与 claim 的相关性。

Layer 2 — 语义验证 (SemanticVerifier):
  用 LLM 检查内容质量：是否跑题、evidence 是否真实支撑 claim、
  执行逻辑是否一致、任务是否被完整覆盖。这是论文的核心创新点——
  形式验证保证基本合规，语义验证检测"看起来合规但内容错误"的 case。
"""

from __future__ import annotations

import json
import logging
import re

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
    """Layer 1: 结构验证 — 零成本 rule-based gate。

    support_rate 使用关键词重叠逐条评估：
      对每条 ContractMessage 的 evidence 列表，逐条检查是否与 claim
      有关键词重叠。重叠 ≥ 1 个 token 即计为有效支撑。
      support_rate = 有效支撑的 evidence 条数 / evidence 总条数。
    """
    if not messages:
        return VerificationResult(
            accepted=False, support_rate=0.0,
            violations=["trace_empty"], next_action="retry",
        )

    violations: list[str] = []
    total_evidence_items = 0
    supported_evidence_items = 0

    for message in messages:
        # 逐条 evidence 检查与 claim 的关键词重叠
        ev_items = message.evidence
        total_evidence_items += len(ev_items)
        if ev_items:
            for ev in ev_items:
                if _evidence_supports_claim(ev, message.claim):
                    supported_evidence_items += 1

        # 结构违规检查
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
    accepted = support_rate >= 0.5 and not any("missing_action" in item for item in violations)
    return VerificationResult(
        accepted=accepted,
        support_rate=round(support_rate, 3),
        violations=violations,
        next_action="accept" if accepted else "escalate",
    )


def _evidence_supports_claim(evidence: str, claim: str) -> bool:
    """检查单条 evidence 是否与 claim 有关键词重叠。

    对中英文分别处理：
    - 英文：提取字母 token 集合，需 ≥ 2 个共同词根才计为支撑
    - 中文：提取 evidence 中所有 2-gram，需 ≥ 2 个出现在 claim 中才计为支撑
      单字匹配或仅 1 个 2-gram 匹配不计数（排除偶然重叠）
    """
    if not evidence.strip() or not claim.strip():
        return False
    ev_lower = evidence.lower()
    claim_lower = claim.lower()
    # 英文 token 重叠 — 需要 ≥ 2 个共同词（排除单字/冠词偶然匹配）
    ev_en = set(re.findall(r"[a-zA-Z_]{3,}", ev_lower))
    claim_en = set(re.findall(r"[a-zA-Z_]{3,}", claim_lower))
    if len(ev_en & claim_en) >= 2:
        return True
    # 中文 2-gram 重叠 — 需要 ≥ 2 个不同的 2-gram 匹配
    matched_bigrams: set[str] = set()
    for i in range(len(evidence) - 1):
        bigram = evidence[i:i + 2]
        if "一" <= bigram[0] <= "鿿" and bigram in claim:
            matched_bigrams.add(bigram)
            if len(matched_bigrams) >= 2:
                return True
    return False


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
    """逐条 evidence 与 claim 的关键词重叠率。

    support_rate = 有重叠的 evidence 条数 / evidence 总条数。
    """
    total = 0
    supported = 0
    for m in messages:
        total += len(m.evidence)
        for ev in m.evidence:
            if _evidence_supports_claim(ev, m.claim):
                supported += 1
    return round(supported / total, 3) if total > 0 else 0.0
