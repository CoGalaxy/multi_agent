from __future__ import annotations

from verifiable_multi_agent.complexity import ComplexityFeatures, InputRequirements
from verifiable_multi_agent.contracts import TaskProfile, Topology
from verifiable_multi_agent.routing_spec import CapabilityNeeds, TaskType, TopologySpec


class QuantitativeRouter:
    def route(
        self,
        profile: TaskProfile,
        features: ComplexityFeatures,
        requirements: InputRequirements,
    ) -> TopologySpec:
        task_type = _task_type(profile, requirements)
        reasons = [
            f"TCI={features.tci:.3f} from horizon={features.horizon:.2f}, "
            f"dependency_depth={features.dependency_depth:.2f}, tool_burden={features.tool_burden:.2f}, "
            f"evidence_burden={features.evidence_burden:.2f}, uncertainty={features.uncertainty:.2f}, risk={features.risk:.2f}",
        ]
        reasons.extend(features.signals)

        blocked = False
        block_reason = None
        if requirements.requires_material and not requirements.material_provided:
            blocked = True
            block_reason = "missing material: task requires given material but no material content was provided"
            reasons.append("blocked_missing_material")

        needs = CapabilityNeeds(
            planning=features.tci >= 0.30 or task_type in {TaskType.COMPARISON, TaskType.MATERIAL_GROUNDED},
            tool_execution=requirements.requires_external_query or requirements.requires_database,
            material_grounding=requirements.requires_material,
            verification=requirements.requires_verification or task_type in {TaskType.COMPARISON, TaskType.MATERIAL_GROUNDED},
            safety_review=features.risk >= 0.50 or requirements.requires_destructive_action,
            synthesis=features.tci >= 0.30 or task_type in {TaskType.COMPARISON, TaskType.MATERIAL_GROUNDED},
        )

        max_review_loops = 1 if needs.safety_review else 0
        max_tool_calls = 3 if needs.tool_execution else 0
        max_nodes = 3  # Executor -> Verifier -> Synthesizer is the minimum executable graph.
        if needs.planning or needs.material_grounding or needs.tool_execution or needs.safety_review:
            max_nodes += 1
        if needs.material_grounding:
            max_nodes += 1
        if needs.tool_execution:
            max_nodes += 1
        if needs.critique:
            max_nodes += 1
        if needs.revision:
            max_nodes += 1
        if needs.safety_review:
            max_nodes += 1
        max_edges = max(0, max_nodes - 1 + max_review_loops)

        return TopologySpec(
            task_type=task_type,
            tci=features.tci,
            capability_needs=needs,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_review_loops=max_review_loops,
            max_tool_calls=max_tool_calls,
            blocked=blocked,
            block_reason=block_reason,
            generation_reasons=reasons,
        )


def topology_from_spec(spec: TopologySpec) -> Topology:
    needs = spec.capability_needs
    if spec.blocked:
        return Topology.SINGLE_AGENT
    if needs.safety_review or needs.critique or needs.revision:
        return Topology.REVIEW_LOOP
    if needs.planning or needs.material_grounding or needs.tool_execution or needs.synthesis:
        return Topology.SUPERVISOR_WORKER
    return Topology.SINGLE_AGENT


def explain_topology_spec(spec: TopologySpec) -> str:
    summary = "; ".join(spec.generation_reasons[:3])
    if spec.blocked:
        return f"QuantRouter blocked normal execution: {spec.block_reason}. Reasons: {summary}"
    return f"QuantRouter selected {topology_from_spec(spec).value} from TopologySpec. Reasons: {summary}"


def _task_type(profile: TaskProfile, requirements: InputRequirements) -> TaskType:
    if requirements.requires_destructive_action or profile.risk >= 0.50:
        return TaskType.HIGH_RISK_OPERATION
    if requirements.requires_material:
        return TaskType.MATERIAL_GROUNDED
    if requirements.requires_external_query or requirements.requires_database:
        return TaskType.TOOL_QUERY
    if requirements.requires_comparison:
        return TaskType.COMPARISON
    if profile.step_count <= 1 and profile.risk < 0.20 and profile.tool_need < 0.20:
        return TaskType.SIMPLE_EXPLANATION
    return TaskType.GENERAL
