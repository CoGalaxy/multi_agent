from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

from verifiable_multi_agent.contracts import TaskProfile, Topology
from verifiable_multi_agent.routing_spec import CapabilityNeeds, TaskType, TopologySpec

if TYPE_CHECKING:
    from verifiable_multi_agent.memory import ProtocolMemory


class InputRequirements(BaseModel):
    """任务输入需求分析 — 从任务文本推断所需能力。"""

    requires_material: bool = False
    material_provided: bool = True
    requires_external_query: bool = False
    requires_database: bool = False
    requires_destructive_action: bool = False
    requires_verification: bool = False
    requires_comparison: bool = False


def infer_input_requirements(task: str) -> InputRequirements:
    """从任务文本推断输入需求和所需能力。"""
    text = task.lower()
    requires_material = _contains_any(
        text,
        [
            "given material", "provided material", "provided context",
            "rag", "corpus", "document collection",
            "根据给定材料", "给定材料", "根据材料", "根据以下材料",
            "以下材料", "材料：", "资料库", "文档库", "检索材料", "rag资料",
        ],
    )
    material_provided = not requires_material or _has_inline_material(task)
    requires_external_query = _contains_any(text, ["query", "search", "lookup", "查询", "搜索", "检索"])
    requires_database = _contains_any(text, ["database", "db", "数据库", "订单"])
    requires_destructive_action = _contains_any(text, ["delete", "remove", "drop", "删除", "清空", "销毁"])
    requires_verification = _contains_any(text, ["verify", "validate", "check", "audit", "验证", "核验", "检查", "审查"])
    requires_comparison = _contains_any(text, ["compare", "comparison", "versus", "vs", "比较", "对比", "差异"])
    return InputRequirements(
        requires_material=requires_material,
        material_provided=material_provided,
        requires_external_query=requires_external_query,
        requires_database=requires_database,
        requires_destructive_action=requires_destructive_action,
        requires_verification=requires_verification,
        requires_comparison=requires_comparison,
    )


# 拓扑升级映射（与 router.py 保持一致）
_TOPOLOGY_UPGRADE = {
    Topology.SINGLE_AGENT: Topology.SUPERVISOR_WORKER,
    Topology.SUPERVISOR_WORKER: Topology.REVIEW_LOOP,
}


class QuantitativeRouter:
    """定量路由器 — 基于任务画像和输入需求生成 TopologySpec。

    使用新的二维画像 (complexity, verifiability) 替代旧的六维 TCI。
    支持可选的 ProtocolMemory 修正：检索历史相似任务，若邻居失败率
    > 0.5 且当前拓扑不是 REVIEW_LOOP，则自动升级一级。
    """

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

        # 使用 complexity 替代旧 TCI 进行能力需求判断
        needs_planning = profile.complexity >= 0.30 or profile.verifiability >= 0.40
        needs_verification = requirements.requires_verification or profile.verifiability >= 0.40
        needs_safety = profile.verifiability >= 0.80 or requirements.requires_destructive_action

        needs = CapabilityNeeds(
            planning=needs_planning or task_type in {TaskType.COMPARISON, TaskType.MATERIAL_GROUNDED},
            tool_execution=requirements.requires_external_query or requirements.requires_database,
            material_grounding=requirements.requires_material,
            verification=needs_verification or task_type in {TaskType.COMPARISON, TaskType.MATERIAL_GROUNDED},
            safety_review=needs_safety,
            synthesis=needs_planning or task_type in {TaskType.COMPARISON, TaskType.MATERIAL_GROUNDED},
        )

        if requirements.requires_comparison:
            reasons.append("signal=comparison")
        if requirements.requires_verification:
            reasons.append("signal=verification_required")
        if requirements.requires_destructive_action:
            reasons.append("signal=destructive_action")

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

        # ── Memory 修正 ─────────────────────────────────────────
        if memory is not None:
            neighbors = memory.query(profile.complexity, profile.verifiability, k=3)
            if neighbors:
                fail_rate = memory.historical_fail_rate(neighbors)
                current_topology = _topology_from_needs(needs)
                if fail_rate > 0.5 and current_topology != Topology.REVIEW_LOOP:
                    upgraded = _TOPOLOGY_UPGRADE[current_topology]
                    # 升级能力需求以匹配新拓扑
                    if upgraded == Topology.SUPERVISOR_WORKER:
                        needs.planning = True
                        needs.synthesis = True
                    elif upgraded == Topology.REVIEW_LOOP:
                        needs.planning = True
                        needs.synthesis = True
                        needs.verification = True
                        max_review_loops = max(max_review_loops, 1)
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
    if requirements.requires_destructive_action or profile.verifiability >= 0.80:
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


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _has_inline_material(task: str) -> bool:
    lower_task = task.lower()
    markers = [
        "：", ":", "\n", "材料如下", "如下材料", "以下材料", "材料：",
        "资料库", "文档库", "rag", "corpus", "AutoGen：", "CAMEL：",
    ]
    return any(marker.lower() in lower_task for marker in markers)


def _topology_from_needs(needs: CapabilityNeeds) -> Topology:
    """从 CapabilityNeeds 推断对应的拓扑（不依赖完整的 TopologySpec）。"""
    if needs.safety_review or needs.critique or needs.revision:
        return Topology.REVIEW_LOOP
    if needs.planning or needs.material_grounding or needs.tool_execution or needs.synthesis:
        return Topology.SUPERVISOR_WORKER
    return Topology.SINGLE_AGENT
