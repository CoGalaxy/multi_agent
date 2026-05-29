from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"


class Topology(str, Enum):
    SINGLE_AGENT = "single_agent"
    SUPERVISOR_WORKER = "supervisor_worker"
    REVIEW_LOOP = "review_loop"


class TaskProfile(BaseModel):
    task: str
    tool_need: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    step_count: int = Field(ge=1)
    risk: float = Field(ge=0.0, le=1.0)

    @property
    def complexity(self) -> float:
        step_score = min(self.step_count / 6, 1.0)
        return round((self.tool_need + self.uncertainty + step_score + self.risk) / 4, 3)


class Budget(BaseModel):
    max_steps: int = 8
    max_messages: int = 12
    remaining_steps: int = 8

    def consume(self) -> None:
        if self.remaining_steps <= 0:
            raise RuntimeError("Budget exhausted")
        self.remaining_steps -= 1


class ContractMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: AgentRole
    subtask: str
    claim: str
    evidence: list[str] = Field(default_factory=list)
    action: str
    uncertainty: float = Field(ge=0.0, le=1.0, default=0.5)
    budget_hint: int = Field(ge=0, default=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_support(self) -> bool:
        return bool(self.evidence) and self.claim.strip() != ""


class VerificationResult(BaseModel):
    accepted: bool
    support_rate: float
    violations: list[str] = Field(default_factory=list)
    next_action: str = "accept"


class AgentTrace(BaseModel):
    task: str
    topology: Topology
    profile: TaskProfile
    messages: list[ContractMessage] = Field(default_factory=list)
    verification: VerificationResult | None = None
    final_answer: str | None = None
