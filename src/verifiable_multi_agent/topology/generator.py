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
                generation_reasons=generation_reasons + ["base_template=blocked_graph"],
                constraints=constraints,
            )

        selected, base_reason = self._base_template(spec)
        generation_reasons.append(base_reason)
        selected, augment_reasons = self._augment_template(selected, spec)
        generation_reasons.extend(augment_reasons)
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

    def _base_template(self, spec: TopologySpec) -> tuple[list[AgentType], str]:
        needs = spec.capability_needs
        if needs.code_testing:
            selected = [
                AgentType.CODER,
                AgentType.TESTER,
                AgentType.REVISER,
                AgentType.VERIFIER,
                AgentType.SYNTHESIZER,
            ]
            if needs.planning:
                selected.insert(0, AgentType.PLANNER)
            return selected, "base_template=code_testing_template"
        if needs.safety_review:
            return [
                AgentType.PLANNER,
                AgentType.EXECUTOR,
                AgentType.SAFETY_VERIFIER,
                AgentType.VERIFIER,
                AgentType.SYNTHESIZER,
            ], "base_template=safety_review_template"
        if needs.material_grounding:
            return [
                AgentType.PLANNER,
                AgentType.RESEARCHER,
                AgentType.EXECUTOR,
                AgentType.VERIFIER,
                AgentType.SYNTHESIZER,
            ], "base_template=evidence_grounded_template"
        if needs.tool_execution:
            return [
                AgentType.PLANNER,
                AgentType.TOOL_EXECUTOR,
                AgentType.EXECUTOR,
                AgentType.VERIFIER,
                AgentType.SYNTHESIZER,
            ], "base_template=tool_assisted_template"
        if needs.planning or needs.synthesis or spec.task_type.value == "comparison":
            return [
                AgentType.PLANNER,
                AgentType.EXECUTOR,
                AgentType.VERIFIER,
                AgentType.SYNTHESIZER,
            ], "base_template=comparison_template"
        return [
            AgentType.EXECUTOR,
            AgentType.VERIFIER,
            AgentType.SYNTHESIZER,
        ], "base_template=simple_template"

    def _augment_template(self, selected: list[AgentType], spec: TopologySpec) -> tuple[list[AgentType], list[str]]:
        needs = spec.capability_needs
        augmented = list(selected)
        reasons: list[str] = []

        if needs.material_grounding and AgentType.RESEARCHER not in augmented:
            _insert_before(augmented, AgentType.RESEARCHER, AgentType.EXECUTOR)
            reasons.append("augment=insert_researcher_before_executor")
        if needs.tool_execution and AgentType.TOOL_EXECUTOR not in augmented:
            _insert_before(augmented, AgentType.TOOL_EXECUTOR, AgentType.EXECUTOR)
            reasons.append("augment=insert_tool_executor_before_executor")
        if needs.critique and AgentType.CRITIC not in augmented:
            _insert_after(augmented, AgentType.CRITIC, AgentType.EXECUTOR)
            reasons.append("augment=insert_critic_after_executor")
        if needs.revision and AgentType.REVISER not in augmented:
            anchor = AgentType.CRITIC if AgentType.CRITIC in augmented else AgentType.EXECUTOR
            _insert_after(augmented, AgentType.REVISER, anchor)
            reasons.append(f"augment=insert_reviser_after_{anchor.value}")
        if needs.safety_review and AgentType.SAFETY_VERIFIER not in augmented:
            _insert_before(augmented, AgentType.SAFETY_VERIFIER, AgentType.VERIFIER)
            reasons.append("augment=insert_safety_verifier_before_verifier")
        if needs.verification and AgentType.VERIFIER not in augmented:
            _insert_before(augmented, AgentType.VERIFIER, AgentType.SYNTHESIZER)
            reasons.append("augment=restore_required_verifier")
        if needs.synthesis and AgentType.SYNTHESIZER not in augmented:
            augmented.append(AgentType.SYNTHESIZER)
            reasons.append("augment=restore_required_synthesizer")

        return augmented, reasons


def _insert_before(nodes: list[AgentType], node: AgentType, anchor: AgentType) -> None:
    if anchor in nodes:
        nodes.insert(nodes.index(anchor), node)
    else:
        nodes.append(node)


def _insert_after(nodes: list[AgentType], node: AgentType, anchor: AgentType) -> None:
    if anchor in nodes:
        nodes.insert(nodes.index(anchor) + 1, node)
    else:
        nodes.append(node)
