from __future__ import annotations

from verifiable_multi_agent.routing_spec import TopologySpec
from verifiable_multi_agent.topology.constraints import constraints_from_spec
from verifiable_multi_agent.topology.graph import AgentEdge, AgentNode, AgentType, CollaborationGraph
from verifiable_multi_agent.topology.templates import NODE_LABELS
from verifiable_multi_agent.topology.validator import validate_graph


class ConstrainedTopologyGenerator:
    def generate(self, spec: TopologySpec) -> CollaborationGraph:
        constraints = constraints_from_spec(spec)
        generation_reasons = list(spec.generation_reasons)
        if spec.blocked:
            return CollaborationGraph(
                blocked=True,
                block_reason=spec.block_reason,
                generation_reasons=generation_reasons + ["compose=blocked_graph"],
                constraints=constraints,
            )

        selected, compose_reasons = self._compose_nodes(spec)
        generation_reasons.extend(compose_reasons)
        nodes = [
            AgentNode(
                id=node_type.value,
                type=node_type,
                label=NODE_LABELS[node_type],
                config={"complexity": spec.complexity, "verifiability": spec.verifiability},
            )
            for node_type in selected
        ]
        edges = [
            AgentEdge(source=nodes[index].id, target=nodes[index + 1].id)
            for index in range(len(nodes) - 1)
        ]
        graph = CollaborationGraph(
            nodes=nodes,
            edges=edges,
            entry_node=nodes[0].id if nodes else None,
            exit_node=nodes[-1].id if nodes else None,
            generation_reasons=generation_reasons,
            constraints=constraints,
        )
        graph.validation_errors = validate_graph(graph)
        return graph

    def _compose_nodes(self, spec: TopologySpec) -> tuple[list[AgentType], list[str]]:
        needs = spec.capability_needs
        selected: list[AgentType] = []
        reasons: list[str] = []

        if needs.planning:
            _append_unique(selected, AgentType.PLANNER, reasons)
        if needs.material_grounding:
            _append_unique(selected, AgentType.RESEARCHER, reasons)
        if needs.tool_execution:
            _append_unique(selected, AgentType.TOOL_EXECUTOR, reasons)

        if needs.code_testing:
            _append_unique(selected, AgentType.CODER, reasons)
            _append_unique(selected, AgentType.TESTER, reasons)
        else:
            _append_unique(selected, AgentType.EXECUTOR, reasons)

        if needs.critique:
            _append_unique(selected, AgentType.CRITIC, reasons)
        if needs.revision or needs.code_testing:
            _append_unique(selected, AgentType.REVISER, reasons)
        if needs.safety_review:
            _append_unique(selected, AgentType.SAFETY_VERIFIER, reasons)

        _append_unique(selected, AgentType.VERIFIER, reasons)
        _append_unique(selected, AgentType.SYNTHESIZER, reasons)
        return selected, reasons


def _append_unique(nodes: list[AgentType], node: AgentType, reasons: list[str]) -> None:
    if node not in nodes:
        nodes.append(node)
        reasons.append(f"compose=add_{node.value}")
