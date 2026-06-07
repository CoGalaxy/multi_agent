from __future__ import annotations

import re

from pydantic import BaseModel, Field

from verifiable_multi_agent.contracts import TaskProfile


class ComplexityFeatures(BaseModel):
    horizon: float = Field(ge=0.0, le=1.0)
    dependency_depth: float = Field(ge=0.0, le=1.0)
    tool_burden: float = Field(ge=0.0, le=1.0)
    evidence_burden: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)

    @property
    def tci(self) -> float:
        value = (
            0.20 * self.horizon
            + 0.20 * self.dependency_depth
            + 0.15 * self.tool_burden
            + 0.15 * self.evidence_burden
            + 0.15 * self.uncertainty
            + 0.15 * self.risk
        )
        return round(value, 3)


class InputRequirements(BaseModel):
    requires_material: bool = False
    material_provided: bool = True
    requires_external_query: bool = False
    requires_database: bool = False
    requires_destructive_action: bool = False
    requires_verification: bool = False
    requires_comparison: bool = False


def infer_complexity_features(task: str, profile: TaskProfile) -> ComplexityFeatures:
    requirements = infer_input_requirements(task)
    text = task.lower()
    signals: list[str] = []

    horizon = min(max(profile.step_count / 4, _count_deliverables(text) / 4), 1.0)
    dependency_depth = 0.0
    if requirements.requires_comparison:
        dependency_depth += 0.35
        signals.append("comparison")
    if requirements.requires_verification:
        dependency_depth += 0.30
        signals.append("verification")
    if _contains_any(text, ["并", "然后", "同时", "and", "then"]):
        dependency_depth += 0.20
        signals.append("ordered_or_multiple_actions")
    dependency_depth = min(dependency_depth, 1.0)

    tool_burden = profile.tool_need
    if requirements.requires_database or requirements.requires_external_query:
        tool_burden = max(tool_burden, 0.75)
        signals.append("tool_query")

    evidence_burden = 0.0
    if requirements.requires_material:
        evidence_burden += 0.60
        signals.append("material_grounding")
    if requirements.requires_comparison:
        evidence_burden += 0.20
    if requirements.requires_verification:
        evidence_burden += 0.20
    evidence_burden = min(evidence_burden, 1.0)

    risk = profile.risk
    if requirements.requires_destructive_action:
        risk = max(risk, 0.90)
        signals.append("destructive_action")

    return ComplexityFeatures(
        horizon=round(horizon, 3),
        dependency_depth=round(dependency_depth, 3),
        tool_burden=round(tool_burden, 3),
        evidence_burden=round(evidence_burden, 3),
        uncertainty=round(profile.uncertainty, 3),
        risk=round(risk, 3),
        signals=signals,
    )


def infer_input_requirements(task: str) -> InputRequirements:
    text = task.lower()
    requires_material = _contains_any(
        text,
        [
            "given material",
            "provided material",
            "provided context",
            "rag",
            "corpus",
            "document collection",
            "根据给定材料",
            "给定材料",
            "根据材料",
            "根据以下材料",
            "以下材料",
            "材料：",
            "资料库",
            "文档库",
            "检索材料",
            "rag资料",
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


def _count_deliverables(text: str) -> int:
    hits = len(re.findall(r"\b(and|then|with)\b", text))
    hits += sum(text.count(item) for item in ["并", "然后", "同时", "给出", "说明", "验证", "比较"])
    return max(1, hits)


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _has_inline_material(task: str) -> bool:
    lower_task = task.lower()
    markers = [
        "：",
        ":",
        "\n",
        "材料如下",
        "如下材料",
        "以下材料",
        "材料：",
        "资料库",
        "文档库",
        "rag",
        "corpus",
        "AutoGen：",
        "CAMEL：",
    ]
    return any(marker.lower() in lower_task for marker in markers)
