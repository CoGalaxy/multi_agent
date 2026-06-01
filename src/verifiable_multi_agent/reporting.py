from __future__ import annotations

from pydantic import BaseModel, Field

from verifiable_multi_agent.contracts import AgentTrace


class ContractReport(BaseModel):
    total_messages: int
    supported_claims: int
    support_rate: float
    evidence_coverage: float
    action_completeness: float
    violations: list[str] = Field(default_factory=list)


def build_contract_report(trace: AgentTrace) -> ContractReport:
    messages = trace.messages
    total = len(messages)
    if total == 0:
        violations = list(trace.verification.violations) if trace.verification else ["trace_empty"]
        return ContractReport(
            total_messages=0,
            supported_claims=0,
            support_rate=0.0,
            evidence_coverage=0.0,
            action_completeness=0.0,
            violations=violations,
        )

    supported = sum(1 for message in messages if message.has_support)
    with_evidence = sum(1 for message in messages if message.evidence)
    with_action = sum(1 for message in messages if message.action.strip())

    violations = list(trace.verification.violations) if trace.verification else []
    for index, message in enumerate(messages, start=1):
        if not message.evidence:
            violations.append(f"{_role_name(message.role.value)}#{index} missing evidence")
        if not message.action.strip():
            violations.append(f"{_role_name(message.role.value)}#{index} missing action")

    return ContractReport(
        total_messages=total,
        supported_claims=supported,
        support_rate=round(supported / total, 3),
        evidence_coverage=round(with_evidence / total, 3),
        action_completeness=round(with_action / total, 3),
        violations=violations,
    )


def format_contract_report(trace: AgentTrace) -> str:
    report = build_contract_report(trace)
    profile = trace.profile
    accepted = trace.verification.accepted if trace.verification else False
    violations = report.model_dump(mode="json")["violations"]
    return "\n".join(
        [
            "[Task Profile]",
            (
                f"tool_need={profile.tool_need:.2f} | "
                f"uncertainty={profile.uncertainty:.2f} | "
                f"risk={profile.risk:.2f} | "
                f"complexity={profile.complexity:.2f}"
            ),
            "",
            "[Topology]",
            f"selected={trace.topology.name}",
            f"reason={trace.topology_reason or ''}",
            "",
            "[Contract Report]",
            f"messages={report.total_messages}",
            f"supported_claims={report.supported_claims}",
            f"support_rate={report.support_rate:.2f}",
            f"evidence_coverage={report.evidence_coverage:.2f}",
            f"action_completeness={report.action_completeness:.2f}",
            f"accepted={accepted}",
            f"violations={violations}",
            "",
            "[Final Answer]",
            trace.final_answer or "",
        ]
    )


def _role_name(role: str) -> str:
    return role[:1].upper() + role[1:]
