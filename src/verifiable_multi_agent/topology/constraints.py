from __future__ import annotations

from pydantic import BaseModel, Field

from verifiable_multi_agent.routing_spec import TopologySpec


class TopologyConstraints(BaseModel):
    max_nodes: int = Field(ge=1)
    max_edges: int = Field(ge=0)
    max_review_loops: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    require_verifier: bool = True
    require_synthesizer: bool = True
    allow_review_loop: bool = True


def constraints_from_spec(spec: TopologySpec) -> TopologyConstraints:
    return TopologyConstraints(
        max_nodes=spec.max_nodes,
        max_edges=spec.max_edges,
        max_review_loops=spec.max_review_loops,
        max_tool_calls=spec.max_tool_calls,
        require_verifier=not spec.blocked,
        require_synthesizer=not spec.blocked,
        allow_review_loop=spec.max_review_loops > 0,
    )
