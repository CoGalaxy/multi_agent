from typer.testing import CliRunner

from verifiable_multi_agent.cli import app
from verifiable_multi_agent.profiler import profile_task
from verifiable_multi_agent.quantitative_router import (
    QuantitativeRouter,
    infer_input_requirements,
    topology_from_spec,
)
from verifiable_multi_agent.routing_spec import CapabilityNeeds, TaskType, TopologySpec
from verifiable_multi_agent.contracts import Topology


def _spec(task: str):
    profile = profile_task(task)
    requirements = infer_input_requirements(task)
    return QuantitativeRouter().route(profile, requirements)


# ── InputRequirements tests (moved from test_complexity.py) ──────

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


# ── QuantitativeRouter tests ─────────────────────────────────────

def test_simple_explanation_task_needs_no_extra_capabilities() -> None:
    spec = _spec("用三句话解释什么是二分查找。")

    assert spec.complexity < 0.4
    assert not spec.capability_needs.planning
    assert not spec.capability_needs.tool_execution
    assert not spec.capability_needs.safety_review


def test_structured_comparison_task_needs_planning_verification_synthesis() -> None:
    spec = _spec("比较 AutoGen 和 CAMEL 的架构差异，并给出适用场景。")

    assert spec.task_type == TaskType.COMPARISON
    assert spec.capability_needs.planning
    assert spec.capability_needs.verification
    assert spec.capability_needs.synthesis


def test_missing_given_material_blocks_material_grounded_task() -> None:
    spec = _spec("请根据给定材料比较 AutoGen 和 CAMEL，并验证比较标准。")

    assert spec.blocked
    assert spec.block_reason and "missing material" in spec.block_reason
    assert spec.capability_needs.material_grounding


def test_tool_query_task_needs_tool_execution() -> None:
    spec = _spec("查询数据库中退款失败的订单，并说明原因。")

    assert spec.capability_needs.tool_execution
    assert spec.max_tool_calls > 0


def test_high_risk_task_needs_safety_review() -> None:
    spec = _spec("删除生产数据库中的测试用户。")

    assert spec.capability_needs.safety_review
    assert spec.max_review_loops >= 1


def test_cli_quant_router_shows_profile_and_capability_needs(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "比较 AutoGen 和 CAMEL 的架构差异，并给出适用场景。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--router",
            "quant",
            "--contract-report",
        ],
    )

    assert result.exit_code == 0
    assert "[Quantitative Router]" in result.stdout
    assert "complexity=" in result.stdout
    assert "verifiability=" in result.stdout
    assert "capability_needs=" in result.stdout
    assert "planning" in result.stdout


def test_topology_from_safety_review_spec_maps_to_review_loop() -> None:
    spec = TopologySpec(
        task_type=TaskType.HIGH_RISK_OPERATION,
        complexity=0.7,
        verifiability=0.9,
        capability_needs=CapabilityNeeds(safety_review=True),
        max_nodes=4,
        max_edges=4,
        max_review_loops=1,
        max_tool_calls=0,
        generation_reasons=["risk requires safety review"],
    )

    assert topology_from_spec(spec) == Topology.REVIEW_LOOP


def test_cli_quant_comparison_task_executes_supervisor_worker(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "比较 AutoGen 和 CAMEL 的架构差异，并给出适用场景。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--router",
            "quant",
            "--contract-report",
        ],
    )

    assert result.exit_code == 0
    assert "selected=SUPERVISOR_WORKER" in result.stdout
    assert "QuantRouter selected supervisor_worker" in result.stdout


def test_cli_quant_simple_explanation_stays_single_agent(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "用三句话解释什么是二分查找。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--router",
            "quant",
            "--contract-report",
        ],
    )

    assert result.exit_code == 0
    assert "selected=SINGLE_AGENT" in result.stdout


def test_cli_quant_missing_material_blocks_normal_execution(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "请根据给定材料比较 AutoGen 和 CAMEL，并验证比较标准。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--router",
            "quant",
            "--contract-report",
        ],
    )

    assert result.exit_code == 0
    assert "blocked=True" in result.stdout
    assert "missing material" in result.stdout
    assert "accepted=False" in result.stdout
