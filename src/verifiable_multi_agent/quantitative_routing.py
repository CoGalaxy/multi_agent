"""Research topic 2a: quantitative routing.

This module is the readable entry point for the routing part of the second
thesis topic. It keeps the quantitative decision chain explicit:

Task text
-> TaskProfile
-> InputRequirements
-> QuantitativeRouter
-> TopologySpec

The generated TopologySpec is then consumed by constrained topology generation.
"""

from __future__ import annotations

from verifiable_multi_agent.contracts import TaskProfile, Topology
from verifiable_multi_agent.profiler import LlmProfiler, profile_task
from verifiable_multi_agent.quantitative_router import (
    InputRequirements,
    QuantitativeRouter,
    explain_topology_spec,
    infer_input_requirements,
    topology_from_spec,
)
from verifiable_multi_agent.routing_spec import CapabilityNeeds, TaskType, TopologySpec

__all__ = [
    "CapabilityNeeds",
    "InputRequirements",
    "LlmProfiler",
    "QuantitativeRouter",
    "TaskProfile",
    "TaskType",
    "Topology",
    "TopologySpec",
    "explain_topology_spec",
    "infer_input_requirements",
    "profile_task",
    "topology_from_spec",
]
