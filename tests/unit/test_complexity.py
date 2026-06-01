from verifiable_multi_agent.complexity import (
    ComplexityFeatures,
    infer_complexity_features,
    infer_input_requirements,
)
from verifiable_multi_agent.profiler import profile_task


def test_tci_formula_uses_weighted_dimensions() -> None:
    features = ComplexityFeatures(
        horizon=0.5,
        dependency_depth=0.4,
        tool_burden=0.3,
        evidence_burden=0.2,
        uncertainty=0.6,
        risk=0.1,
    )

    assert features.tci == 0.36


def test_simple_explanation_has_low_tci() -> None:
    task = "用三句话解释什么是二分查找。"
    profile = profile_task(task)
    features = infer_complexity_features(task, profile)

    assert features.tci < 0.25
    assert features.risk == 0.0
    assert features.tool_burden == 0.0


def test_given_material_without_inline_content_is_required_but_missing() -> None:
    requirements = infer_input_requirements("请根据给定材料比较 AutoGen 和 CAMEL，并验证比较标准。")

    assert requirements.requires_material
    assert not requirements.material_provided
    assert requirements.requires_comparison
    assert requirements.requires_verification


def test_given_material_with_inline_content_is_not_missing() -> None:
    requirements = infer_input_requirements(
        "请根据给定材料比较 AutoGen 和 CAMEL。材料如下：AutoGen 支持多 Agent；CAMEL 强调角色扮演。"
    )

    assert requirements.requires_material
    assert requirements.material_provided
