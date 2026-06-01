from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from verifiable_multi_agent.topology.constraints import TopologyConstraints


class AgentType(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    TOOL_EXECUTOR = "tool_executor"
    EXECUTOR = "executor"
    CODER = "coder"
    TESTER = "tester"
    CRITIC = "critic"
    REVISER = "reviser"
    SAFETY_VERIFIER = "safety_verifier"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"


class AgentNode(BaseModel):
    id: str
    type: AgentType
    label: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class AgentEdge(BaseModel):
    source: str
    target: str
    label: str | None = None


class CollaborationGraph(BaseModel):
    nodes: list[AgentNode] = Field(default_factory=list)
    edges: list[AgentEdge] = Field(default_factory=list)
    entry_node: str | None = None
    exit_node: str | None = None
    blocked: bool = False
    block_reason: str | None = None
    generation_reasons: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    constraints: TopologyConstraints

    def node_ids(self) -> set[str]:
        return {node.id for node in self.nodes}

    def node_types(self) -> list[str]:
        return [node.type.value for node in self.nodes]
