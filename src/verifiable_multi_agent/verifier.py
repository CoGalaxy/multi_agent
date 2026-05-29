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
    accepted = support_rate >= 0.8 and not any("missing_action" in item for item in violations)
    return VerificationResult(
        accepted=accepted,
        support_rate=round(support_rate, 3),
        violations=violations,
        next_action="accept" if accepted else "escalate",
    )
