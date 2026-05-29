from __future__ import annotations

from verifiable_multi_agent.contracts import TaskProfile, Topology


def select_topology(profile: TaskProfile) -> Topology:
    if profile.risk >= 0.5 or profile.complexity >= 0.55:
        return Topology.REVIEW_LOOP
    if profile.complexity >= 0.25 or profile.step_count >= 3:
        return Topology.SUPERVISOR_WORKER
    return Topology.SINGLE_AGENT
