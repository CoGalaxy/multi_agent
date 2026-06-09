from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    SIMPLE_EXPLANATION = "simple_explanation"
    COMPARISON = "comparison"
    MATERIAL_GROUNDED = "material_grounded"
    TOOL_QUERY = "tool_query"
    HIGH_RISK_OPERATION = "high_risk_operation"
    GENERAL = "general"


class CapabilityNeeds(BaseModel):
    planning: bool = False
    tool_execution: bool = False
    material_grounding: bool = False
    verification: bool = False
    code_testing: bool = False
    critique: bool = False
    revision: bool = False
    safety_review: bool = False
    synthesis: bool = False


class TopologySpec(BaseModel):
    task_type: TaskType
    complexity: float = Field(ge=0.0, le=1.0)
    verifiability: float = Field(ge=0.0, le=1.0)
    capability_needs: CapabilityNeeds
    max_nodes: int = Field(ge=1)
    max_edges: int = Field(ge=0)
    max_review_loops: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    blocked: bool = False
    block_reason: str | None = None
    generation_reasons: list[str] = Field(default_factory=list)
