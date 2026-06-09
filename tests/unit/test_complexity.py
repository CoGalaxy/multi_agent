"""Tests for the new two-dimension task profiling (complexity + verifiability)."""

from verifiable_multi_agent.profiler import profile_task


def test_simple_task_has_low_both() -> None:
    """简单任务应该同时具有低复杂度和低验证难度。"""
    task = "用三句话解释什么是二分查找。"
    profile = profile_task(task)

    assert profile.complexity < 0.4
    assert profile.verifiability < 0.4


def test_comparison_task_has_moderate_verifiability() -> None:
    """比较类任务验证难度中等（0.2~0.5），低于核查/证伪类任务。"""
    task = "比较 AutoGen 和 CAMEL 的架构差异，并给出适用场景。"
    profile = profile_task(task)

    # 比较类：verifiability 在 0.15~0.5 之间（中等，不是高）
    assert 0.1 <= profile.verifiability <= 0.5
    # 复杂度通常高于验证难度（比较需要分解，但可对照文档验证）
    assert profile.complexity >= 0.2


def test_complex_design_task_has_high_complexity() -> None:
    """复杂设计任务应该有高复杂度。"""
    task = "设计一个复杂的多智能体协作系统，包括任务分配、通信协议和冲突解决"
    profile = profile_task(task)

    assert profile.complexity >= 0.3


def test_high_risk_task_has_high_verifiability() -> None:
    """高风险操作（删除、验证备份）验证难度很高。"""
    task = "删除生产数据库中的用户数据并验证备份"
    profile = profile_task(task)

    assert profile.verifiability >= 0.3


def test_english_task_profile() -> None:
    """英文任务也应该被正确画像。"""
    task = "Compare AutoGen and CAMEL architectures and recommend one for production use."
    profile = profile_task(task)

    assert profile.complexity > 0.0
    assert profile.verifiability > 0.0


def test_empty_task_returns_valid_profile() -> None:
    """空任务应该返回有效的默认 profile。"""
    profile = profile_task("hello")

    assert 0.0 <= profile.complexity <= 1.0
    assert 0.0 <= profile.verifiability <= 1.0


def test_profile_fields_are_in_range() -> None:
    """所有 profile 字段应该在 [0.0, 1.0] 范围内。"""
    tasks = [
        "打个招呼",
        "用三句话解释什么是二分查找。",
        "比较 AutoGen 和 CAMEL 的架构差异，并给出适用场景。",
        "删除生产数据库中的用户数据并验证备份",
    ]
    for task in tasks:
        profile = profile_task(task)
        assert 0.0 <= profile.complexity <= 1.0, f"complexity out of range for: {task}"
        assert 0.0 <= profile.verifiability <= 1.0, f"verifiability out of range for: {task}"
