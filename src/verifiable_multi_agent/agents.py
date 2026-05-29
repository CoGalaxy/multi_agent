from __future__ import annotations

from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.contracts import AgentRole, ContractMessage


class RuleBasedAgent:
    def __init__(self, role: AgentRole, backend: LlmBackend | None = None) -> None:
        self.role = role
        self.backend = backend

    def run(self, task: str, context: list[ContractMessage]) -> ContractMessage:
        if self.role == AgentRole.PLANNER:
            claim = f"The task can be handled with {max(1, min(3, len(task) // 60 + 1))} focused step(s)."
            if self.backend:
                claim = self.backend.complete("You are a planning agent. Return one concise plan claim.", task)
            return ContractMessage(
                role=self.role,
                subtask="Decompose task and choose execution plan",
                claim=claim,
                evidence=["Task profile and requested deliverable were inspected."],
                action="Create an execution checklist and hand it to the executor.",
                uncertainty=0.25,
            )
        if self.role == AgentRole.EXECUTOR:
            plan = _latest_claim(context, AgentRole.PLANNER)
            claim = f"Candidate answer for task: {task}"
            if self.backend:
                claim = self.backend.complete(
                    "You are an execution agent. Draft a concise candidate answer.",
                    f"Task: {task}\nPlan: {plan or 'direct execution'}",
                )
            return ContractMessage(
                role=self.role,
                subtask="Produce candidate solution",
                claim=claim,
                evidence=[plan or "Direct task statement is available."],
                action="Draft the answer and expose assumptions for verification.",
                uncertainty=0.35,
            )
        if self.role == AgentRole.VERIFIER:
            unsupported = [message.id for message in context if not message.has_support]
            return ContractMessage(
                role=self.role,
                subtask="Check contract support",
                claim="All prior claims are supported." if not unsupported else "Some claims lack support.",
                evidence=["Checked each contract message for claim and evidence fields."],
                action="Accept trace." if not unsupported else "Request local retry for unsupported messages.",
                uncertainty=0.15 if not unsupported else 0.65,
                metadata={"unsupported_message_ids": unsupported},
            )
        claim = _latest_claim(context, AgentRole.EXECUTOR) or task
        if self.backend:
            claim = self.backend.complete(
                "You are a synthesis agent. Produce a concise final answer.",
                "\n".join(message.claim for message in context),
            )
        return ContractMessage(
            role=self.role,
            subtask="Synthesize final answer",
            claim=claim,
            evidence=[message.claim for message in context if message.has_support],
            action="Return concise final response with trace summary.",
            uncertainty=0.2,
        )


def _latest_claim(messages: list[ContractMessage], role: AgentRole) -> str | None:
    for message in reversed(messages):
        if message.role == role:
            return message.claim
    return None
