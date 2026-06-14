"""Research topic 2: constrained multi-agent topology generation.

This module gathers the quant route used by the thesis main line:
TaskProfile -> QuantitativeRouter -> TopologySpec -> CollaborationGraph.
"""

from __future__ import annotations

from verifiable_multi_agent.quantitative_routing import (
    InputRequirements,
    QuantitativeRouter,
    CapabilityNeeds,
    TaskType,
    TopologySpec,
    explain_topology_spec,
    infer_input_requirements,
    topology_from_spec,
)
from verifiable_multi_agent.topology.generator import ConstrainedTopologyGenerator
from verifiable_multi_agent.topology.graph import AgentEdge, AgentNode, AgentType, CollaborationGraph

__all__ = [
    "AgentEdge",
    "AgentNode",
    "AgentType",
    "CapabilityNeeds",
    "CollaborationGraph",
    "ConstrainedTopologyGenerator",
    "InputRequirements",
    "QuantitativeRouter",
    "TaskType",
    "TopologySpec",
    "explain_topology_spec",
    "infer_input_requirements",
    "topology_from_spec",
]
