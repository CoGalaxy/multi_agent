from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from verifiable_multi_agent.contracts import TaskProfile, Topology
from verifiable_multi_agent.routing_spec import CapabilityNeeds, TaskType, TopologySpec

if TYPE_CHECKING:
    from verifiable_multi_agent.memory import ProtocolMemory


class InputRequirements(BaseModel):
    """Task-side signals inferred before quantitative routing."""

    requires_material: bool = False
    material_provided: bool = True
    requires_external_query: bool = False
    requires_database: bool = False
    requires_destructive_action: bool = False
    requires_verification: bool = False
    requires_comparison: bool = False
    requires_code_generation: bool = False
    requires_code_testing: bool = False
    requires_revision: bool = False
    requires_critique: bool = False
    requires_research: bool = False
    requires_safety_review: bool = False


def infer_input_requirements(task: str) -> InputRequirements:
    """Infer explicit task requirements used by the quantitative router."""
    text = task.lower()
    requires_material = _contains_any(
        text,
        [
            "given material", "provided material", "provided context",
            "rag", "corpus", "document collection",
            "\u6839\u636e\u7ed9\u5b9a\u6750\u6599", "\u7ed9\u5b9a\u6750\u6599",
            "\u6839\u636e\u6750\u6599", "\u6839\u636e\u4ee5\u4e0b\u6750\u6599",
            "\u4ee5\u4e0b\u6750\u6599", "\u6750\u6599\uff1a",
            "\u8d44\u6599\u5e93", "\u6587\u6863\u5e93", "\u68c0\u7d22\u6750\u6599",
            "rag\u8d44\u6599",
        ],
    )
    material_provided = not requires_material or _has_inline_material(task)
    requires_external_query = _contains_any(
        text,
        [
            "query", "lookup", "web search", "internet search", "search for",
            "\u67e5\u8be2", "\u641c\u7d22", "\u68c0\u7d22",
        ],
    )
    requires_database = _contains_any(
        text,
        ["database", "db", "\u6570\u636e\u5e93", "\u8ba2\u5355"],
    )
    requires_destructive_action = _contains_any(
        text,
        ["delete", "remove", "drop", "\u5220\u9664", "\u6e05\u7a7a", "\u9500\u6bc1"],
    )
    requires_verification = _contains_any(
        text,
        ["verify", "validate", "check", "audit", "\u9a8c\u8bc1", "\u6821\u9a8c", "\u68c0\u67e5", "\u5ba1\u67e5"],
    )
    requires_comparison = _contains_any(
        text,
        ["compare", "comparison", "versus", "vs", "\u6bd4\u8f83", "\u5bf9\u6bd4", "\u5dee\u5f02"],
    )
    requires_code_generation = _contains_any(
        text,
        [
            "implement", "write code", "code", "function", "class", "api",
            "\u5b9e\u73b0", "\u7f16\u5199\u4ee3\u7801", "\u5199\u4ee3\u7801",
            "\u51fd\u6570", "\u7c7b", "\u63a5\u53e3",
        ],
    )
    requires_code_testing = _contains_any(
        text,
        [
            "test", "unit test", "edge case", "pytest",
            "\u6d4b\u8bd5", "\u5355\u5143\u6d4b\u8bd5",
            "\u8fb9\u754c\u60c5\u51b5", "\u7528\u4f8b",
        ],
    )
    requires_revision = _contains_any(
        text,
        [
            "revise", "revision", "modify", "improve", "refine", "feedback",
            "\u4fee\u6539", "\u6539\u8fdb", "\u4fee\u8ba2",
            "\u6839\u636e\u53cd\u9988", "\u5b8c\u5584",
        ],
    )
    requires_critique = _contains_any(
        text,
        [
            "critique", "criticize", "find issues", "point out", "argue against",
            "\u6279\u5224", "\u627e\u95ee\u9898", "\u6307\u51fa\u4e0d\u8db3",
            "\u53cd\u9a73", "\u6311\u9519",
        ],
    )
    requires_research = _contains_any(
        text,
        [
            "research", "investigate", "docs", "documentation", "reference material",
            "\u8c03\u7814", "\u7814\u7a76", "\u6587\u6863",
            "\u8d44\u6599", "\u68c0\u7d22\u8d44\u6599",
        ],
    )
    requires_safety_review = _contains_any(
        text,
        [
            "safety", "security", "risk", "compliance", "permission", "privacy", "production",
            "\u5b89\u5168", "\u98ce\u9669", "\u5408\u89c4",
            "\u6743\u9650", "\u9690\u79c1", "\u751f\u4ea7",
        ],
    )
    return InputRequirements(
        requires_material=requires_material,
        material_provided=material_provided,
        requires_external_query=requires_external_query,
        requires_database=requires_database,
        requires_destructive_action=requires_destructive_action,
        requires_verification=requires_verification,
        requires_comparison=requires_comparison,
        requires_code_generation=requires_code_generation,
        requires_code_testing=requires_code_testing,
        requires_revision=requires_revision,
        requires_critique=requires_critique,
        requires_research=requires_research,
        requires_safety_review=requires_safety_review,
    )


# Topology upgrade mapping used by route memory correction.
_TOPOLOGY_UPGRADE = {
    Topology.SINGLE_AGENT: Topology.SUPERVISOR_WORKER,
    Topology.SUPERVISOR_WORKER: Topology.REVIEW_LOOP,
}


class QuantitativeRouter:
    """Quantitative router that turns TaskProfile + requirements into TopologySpec."""

    def route(
        self,
        profile: TaskProfile,
        requirements: InputRequirements,
        memory: "ProtocolMemory | None" = None,
    ) -> TopologySpec:
        task_type = _task_type(profile, requirements)
        reasons = [
            f"complexity={profile.complexity:.2f}, "
            f"verifiability={profile.verifiability:.2f}",
        ]

        blocked = False
        block_reason = None
        if requirements.requires_material and not requirements.material_provided:
            blocked = True
            block_reason = "missing material: task requires given material but no material content was provided"
            reasons.append("blocked_missing_material")

        explicit_code = requirements.requires_code_generation or requirements.requires_code_testing
        explicit_review = requirements.requires_revision or requirements.requires_critique
        explicit_tool = requirements.requires_external_query or requirements.requires_database
        explicit_material = requirements.requires_material or requirements.requires_research
        needs_planning = (
            profile.complexity >= 0.30
            or profile.verifiability >= 0.40
            or requirements.requires_comparison
            or explicit_material
            or explicit_tool
            or explicit_code
            or explicit_review
            or requirements.requires_safety_review
            or requirements.requires_destructive_action
        )
        needs_verification = (
            requirements.requires_verification
            or profile.verifiability >= 0.40
            or requirements.requires_comparison
            or explicit_material
            or explicit_code
            or explicit_review
            or requirements.requires_safety_review
        )
        needs_safety = (
            profile.verifiability >= 0.80
            or requirements.requires_destructive_action
            or requirements.requires_safety_review
        )

        needs = CapabilityNeeds(
            planning=needs_planning,
            tool_execution=explicit_tool,
            material_grounding=explicit_material,
            verification=needs_verification,
            code_testing=explicit_code,
            critique=requirements.requires_critique,
            revision=requirements.requires_revision,
            safety_review=needs_safety,
            synthesis=needs_planning
            or requirements.requires_comparison
            or explicit_material
            or explicit_tool
            or explicit_code
            or explicit_review
            or needs_safety,
        )

        _append_signal_reasons(requirements, reasons)

        max_review_loops = 1 if needs.safety_review or needs.critique or needs.revision else 0
        max_tool_calls = 3 if needs.tool_execution else 0
        max_nodes = _estimate_node_count(needs)
        max_edges = max(0, max_nodes - 1 + max_review_loops)

        if memory is not None:
            neighbors = memory.query(profile.complexity, profile.verifiability, k=3)
            if neighbors:
                fail_rate = memory.historical_fail_rate(neighbors)
                current_topology = _matrix_topology(profile.complexity, profile.verifiability)
                if fail_rate > 0.5 and current_topology != Topology.REVIEW_LOOP:
                    upgraded = _TOPOLOGY_UPGRADE[current_topology]
                    if upgraded == Topology.SUPERVISOR_WORKER:
                        needs.planning = True
                        needs.synthesis = True
                    elif upgraded == Topology.REVIEW_LOOP:
                        needs.planning = True
                        needs.synthesis = True
                        needs.verification = True
                        max_review_loops = max(max_review_loops, 1)
                    max_nodes = _estimate_node_count(needs)
                    max_edges = max(0, max_nodes - 1 + max_review_loops)
                    reasons.append(
                        f"memory correction: historical fail rate={fail_rate:.3f}, "
                        f"upgraded to {upgraded.value.upper()}"
                    )

        return TopologySpec(
            task_type=task_type,
            complexity=profile.complexity,
            verifiability=profile.verifiability,
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
    """Map TopologySpec to the three thesis macro topology levels."""
    if spec.blocked:
        return Topology.SINGLE_AGENT
    needs = spec.capability_needs
    if spec.verifiability >= 0.4:
        return Topology.REVIEW_LOOP
    if needs.safety_review or needs.critique or needs.revision or spec.max_review_loops > 0:
        return Topology.REVIEW_LOOP
    if (
        needs.planning
        or needs.material_grounding
        or needs.tool_execution
        or needs.verification
        or needs.synthesis
    ):
        return Topology.SUPERVISOR_WORKER
    if spec.complexity < 0.4 and spec.verifiability < 0.4:
        return Topology.SINGLE_AGENT
    if spec.complexity >= 0.4 and spec.verifiability < 0.4:
        return Topology.SUPERVISOR_WORKER
    return Topology.REVIEW_LOOP


def explain_topology_spec(spec: TopologySpec) -> str:
    summary = "; ".join(spec.generation_reasons[:3])
    if spec.blocked:
        return f"QuantRouter blocked normal execution: {spec.block_reason}. Reasons: {summary}"
    return f"QuantRouter selected {topology_from_spec(spec).value} from TopologySpec. Reasons: {summary}"


def _task_type(profile: TaskProfile, requirements: InputRequirements) -> TaskType:
    if (
        requirements.requires_destructive_action
        or requirements.requires_safety_review
        or profile.verifiability >= 0.80
    ):
        return TaskType.HIGH_RISK_OPERATION
    if requirements.requires_material:
        return TaskType.MATERIAL_GROUNDED
    if requirements.requires_external_query or requirements.requires_database:
        return TaskType.TOOL_QUERY
    if requirements.requires_comparison:
        return TaskType.COMPARISON
    if profile.complexity < 0.25 and profile.verifiability < 0.25:
        return TaskType.SIMPLE_EXPLANATION
    return TaskType.GENERAL


def _append_signal_reasons(requirements: InputRequirements, reasons: list[str]) -> None:
    if requirements.requires_comparison:
        reasons.append("signal=comparison")
    if requirements.requires_verification:
        reasons.append("signal=verification_required")
    if requirements.requires_material:
        reasons.append("signal=material_grounding")
    if requirements.requires_research:
        reasons.append("signal=research_required")
    if requirements.requires_external_query or requirements.requires_database:
        reasons.append("signal=tool_execution")
    if requirements.requires_code_generation or requirements.requires_code_testing:
        reasons.append("signal=code_testing")
    if requirements.requires_revision:
        reasons.append("signal=revision_required")
    if requirements.requires_critique:
        reasons.append("signal=critique_required")
    if requirements.requires_safety_review:
        reasons.append("signal=safety_review")
    if requirements.requires_destructive_action:
        reasons.append("signal=destructive_action")


def _estimate_node_count(needs: CapabilityNeeds) -> int:
    nodes = 3  # Coder/Executor -> Verifier -> Synthesizer.
    if needs.planning:
        nodes += 1
    if needs.material_grounding:
        nodes += 1
    if needs.tool_execution:
        nodes += 1
    if needs.code_testing:
        nodes += 1  # Tester; Coder replaces Executor.
        nodes += 1  # Reviser is part of the constrained code path.
    if needs.critique:
        nodes += 1
    if needs.revision and not needs.code_testing:
        nodes += 1
    if needs.safety_review:
        nodes += 1
    return nodes


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _has_inline_material(task: str) -> bool:
    lower_task = task.lower()
    markers = [
        "\uff1a", ":", "\n", "\u6750\u6599\u5982\u4e0b",
        "\u5982\u4e0b\u6750\u6599", "\u4ee5\u4e0b\u6750\u6599",
        "\u6750\u6599\uff1a", "\u8d44\u6599\u5e93", "\u6587\u6863\u5e93",
        "rag", "corpus", "autogen\uff1a", "camel\uff1a",
    ]
    return any(marker.lower() in lower_task for marker in markers)


def _matrix_topology(complexity: float, verifiability: float) -> Topology:
    """Two-dimensional decision matrix shared with topology_from_spec."""
    if complexity < 0.4 and verifiability < 0.4:
        return Topology.SINGLE_AGENT
    if complexity >= 0.4 and verifiability < 0.4:
        return Topology.SUPERVISOR_WORKER
    return Topology.REVIEW_LOOP
