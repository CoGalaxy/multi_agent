from typer.testing import CliRunner

import verifiable_multi_agent.cli as cli_module
from verifiable_multi_agent.cli import app
from verifiable_multi_agent.contracts import Topology
from verifiable_multi_agent.profiler import profile_task
from verifiable_multi_agent.quantitative_router import (
    QuantitativeRouter,
    infer_input_requirements,
    topology_from_spec,
)
from verifiable_multi_agent.routing_spec import CapabilityNeeds, TaskType, TopologySpec

from tests.fake_llm import FakeLlmBackend


def _spec(task: str):
    profile = profile_task(task)
    requirements = infer_input_requirements(task)
    return QuantitativeRouter().route(profile, requirements)


def test_given_material_without_inline_content_is_required_but_missing() -> None:
    requirements = infer_input_requirements(
        "Compare AutoGen and CAMEL based on the given material, and verify the criteria."
    )

    assert requirements.requires_material
    assert not requirements.material_provided
    assert requirements.requires_comparison
    assert requirements.requires_verification


def test_given_material_with_inline_content_is_not_missing() -> None:
    requirements = infer_input_requirements(
        "Compare AutoGen and CAMEL based on the given material: "
        "AutoGen supports multi-agent chat; CAMEL focuses on role play."
    )

    assert requirements.requires_material
    assert requirements.material_provided


def test_input_requirements_detect_dynamic_topology_signals() -> None:
    requirements = infer_input_requirements(
        "Implement a class API, add unit tests for edge cases, critique the result, "
        "revise it based on feedback, research the docs, and run a safety review."
    )

    assert requirements.requires_code_generation
    assert requirements.requires_code_testing
    assert requirements.requires_critique
    assert requirements.requires_revision
    assert requirements.requires_research
    assert requirements.requires_safety_review


def test_simple_explanation_task_needs_no_extra_capabilities() -> None:
    spec = _spec("Explain binary search in three short sentences.")

    assert spec.complexity < 0.4
    assert not spec.capability_needs.planning
    assert not spec.capability_needs.tool_execution
    assert not spec.capability_needs.safety_review


def test_structured_comparison_task_needs_planning_verification_synthesis() -> None:
    spec = _spec("Compare the architecture differences between AutoGen and CAMEL and give suitable scenarios.")

    assert spec.task_type == TaskType.COMPARISON
    assert spec.capability_needs.planning
    assert spec.capability_needs.verification
    assert spec.capability_needs.synthesis
    assert "signal=comparison" in spec.generation_reasons


def test_missing_given_material_blocks_material_grounded_task() -> None:
    spec = _spec("Compare AutoGen and CAMEL based on the given material, and verify the criteria.")

    assert spec.blocked
    assert spec.block_reason and "missing material" in spec.block_reason
    assert spec.capability_needs.material_grounding


def test_tool_query_task_needs_tool_execution() -> None:
    spec = _spec("Query the database for failed refund orders and explain the reason.")

    assert spec.capability_needs.tool_execution
    assert spec.max_tool_calls > 0
    assert "signal=tool_execution" in spec.generation_reasons


def test_high_risk_task_needs_safety_review() -> None:
    spec = _spec("Delete test users from the production database.")

    assert spec.capability_needs.safety_review
    assert spec.max_review_loops >= 1
    assert "signal=destructive_action" in spec.generation_reasons


def test_code_testing_task_maps_to_code_capabilities() -> None:
    spec = _spec("Implement a function and add unit tests for edge cases.")

    assert spec.capability_needs.planning
    assert spec.capability_needs.code_testing
    assert spec.capability_needs.verification
    assert spec.max_nodes >= 6
    assert "signal=code_testing" in spec.generation_reasons


def test_revision_and_critique_task_maps_to_review_capabilities() -> None:
    spec = _spec("Critique this answer, point out issues, and revise it based on feedback.")

    assert spec.capability_needs.critique
    assert spec.capability_needs.revision
    assert spec.max_review_loops >= 1
    assert "signal=critique_required" in spec.generation_reasons
    assert "signal=revision_required" in spec.generation_reasons


def test_research_task_maps_to_material_grounding_without_blocking() -> None:
    spec = _spec("Research the documentation and summarize the differences.")

    assert spec.capability_needs.material_grounding
    assert not spec.blocked
    assert "signal=research_required" in spec.generation_reasons


def _patch_backend(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "_build_backend", lambda *args, **kwargs: (FakeLlmBackend(), None))


def test_cli_quant_router_shows_profile_and_capability_needs(tmp_path, monkeypatch) -> None:
    _patch_backend(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "Compare the architecture differences between AutoGen and CAMEL and give suitable scenarios.",
            "--backend",
            "vllm",
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


def test_cli_quant_comparison_task_routes_by_capability_needs(tmp_path, monkeypatch) -> None:
    _patch_backend(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "Compare the architecture differences between AutoGen and CAMEL and give suitable scenarios.",
            "--backend",
            "vllm",
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


def test_cli_quant_simple_explanation_stays_single_agent(tmp_path, monkeypatch) -> None:
    _patch_backend(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "Explain binary search in three short sentences.",
            "--backend",
            "vllm",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--router",
            "quant",
            "--contract-report",
        ],
    )

    assert result.exit_code == 0
    assert "selected=SINGLE_AGENT" in result.stdout


def test_cli_quant_missing_material_blocks_normal_execution(tmp_path, monkeypatch) -> None:
    _patch_backend(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "Compare AutoGen and CAMEL based on the given material, and verify the criteria.",
            "--backend",
            "vllm",
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
