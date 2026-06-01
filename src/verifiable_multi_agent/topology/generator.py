from __future__ import annotations

from verifiable_multi_agent.routing_spec import TopologySpec
from verifiable_multi_agent.topology.constraints import constraints_from_spec
from verifiable_multi_agent.topology.graph import AgentEdge, AgentNode, AgentType, CollaborationGraph
from verifiable_multi_agent.topology.templates import CODE_TESTING_ORDER, DEFAULT_ORDER, NODE_LABELS
from verifiable_multi_agent.topology.validator import validate_graph


class ConstrainedTopologyGenerator:
    def generate(self, spec: TopologySpec) -> CollaborationGraph:
        constraints = constraints_from_spec(spec)
        if spec.blocked:
            return CollaborationGraph(
                blocked=True,
                block_reason=spec.block_reason,
                generation_reasons=spec.generation_reasons,
                constraints=constraints,
            )

        order = CODE_TESTING_ORDER if spec.capability_needs.code_testing else DEFAULT_ORDER
        selected = self._select_nodes(spec, order)
        nodes = [
            AgentNode(id=node_type.value, type=node_type, label=NODE_LABELS[node_type], config={"tci": spec.tci})
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
            generation_reasons=spec.generation_reasons,
            constraints=constraints,
        )
        graph.validation_errors = validate_graph(graph)
        return graph

    def _select_nodes(self, spec: TopologySpec, order: list[AgentType]) -> list[AgentType]:
        needs = spec.capability_needs
        if needs.code_testing:
            selected = [AgentType.CODER, AgentType.TESTER, AgentType.REVISER, AgentType.VERIFIER, AgentType.SYNTHESIZER]
            if needs.planning:
                selected.insert(0, AgentType.PLANNER)
            return [node for node in order if node in selected]

        selected = {AgentType.EXECUTOR, AgentType.VERIFIER, AgentType.SYNTHESIZER}
        if needs.planning or needs.material_grounding or needs.tool_execution or needs.safety_review:
            selected.add(AgentType.PLANNER)
        if needs.material_grounding:
            selected.add(AgentType.RESEARCHER)
        if needs.tool_execution:
            selected.add(AgentType.TOOL_EXECUTOR)
        if needs.critique:
            selected.add(AgentType.CRITIC)
        if needs.revision:
            selected.add(AgentType.REVISER)
        if needs.safety_review:
            selected.add(AgentType.SAFETY_VERIFIER)
        return [node for node in order if node in selected]
