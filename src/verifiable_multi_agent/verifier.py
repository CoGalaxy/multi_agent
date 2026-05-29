"""
合约验证器 — 检查协作过程的结构合规性。

当前验证是 rule-based 的：检查 claim 是否有 evidence 支撑、
action 是否非空、高不确定性是否伴随预算提示。这属于"形式验证"——
不判断内容是否正确，只判断步骤是否完整。

形式验证 + 语义验证的对比是论文中可以展开的讨论点：
- 形式验证：零成本、可复现，但无法检测"看似合规但内容错误"
- 语义验证（后续用本地模型）：能检测内容错误，但有额外推理成本
"""

from __future__ import annotations

from verifiable_multi_agent.contracts import ContractMessage, VerificationResult


def verify_contracts(messages: list[ContractMessage]) -> VerificationResult:
    if not messages:
        return VerificationResult(
            accepted=False,
            support_rate=0.0,
            violations=["trace_empty"],
            next_action="retry",
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
    # 接受条件：≥80% 的消息有支撑，且没有任何消息缺失 action
    accepted = support_rate >= 0.8 and not any("missing_action" in item for item in violations)
    return VerificationResult(
        accepted=accepted,
        support_rate=round(support_rate, 3),
        violations=violations,
        next_action="accept" if accepted else "escalate",
    )
