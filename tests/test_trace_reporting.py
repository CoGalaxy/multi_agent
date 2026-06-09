import json
from pathlib import Path

from typer.testing import CliRunner

from verifiable_multi_agent.cli import app
from verifiable_multi_agent.contracts import (
    AgentRole,
    AgentTrace,
    ContractMessage,
    TaskProfile,
    Topology,
    VerificationResult,
)
from verifiable_multi_agent.reporting import build_contract_report, format_contract_report
from verifiable_multi_agent.trace import build_run_trace, save_run_trace


def _sample_trace() -> AgentTrace:
    trace = AgentTrace(
        task="比较两个多智能体框架，并验证比较标准。",
        topology=Topology.SUPERVISOR_WORKER,
        profile=TaskProfile(
            task="比较两个多智能体框架，并验证比较标准。",
            complexity=0.55,
            verifiability=0.75,
        ),
        topology_reason="complexity=0.55, verifiability=0.75 → SUPERVISOR_WORKER",
        final_answer="AutoGen 更适合复杂工具协作，CAMEL 更适合轻量基线。",
    )
    trace.messages = [
        ContractMessage(
            role=AgentRole.PLANNER,
            subtask="定义比较标准",
            claim="需要从易用性和扩展性比较。",
            evidence=["task asks for comparison"],
            action="列出比较维度。",
        ),
        ContractMessage(
            role=AgentRole.EXECUTOR,
            subtask="生成对比答案",
            claim="AutoGen 扩展性更强。",
            evidence=["planner criteria"],
            action="生成对比表。",
        ),
        ContractMessage(
            role=AgentRole.EXECUTOR,
            subtask="补充结论",
            claim="CAMEL 更适合作为轻量基线。",
            evidence=[],
            action="补充结论。",
        ),
        ContractMessage(
            role=AgentRole.VERIFIER,
            subtask="验证覆盖度",
            claim="比较覆盖主要维度。",
            evidence=["audited prior messages"],
            action="确认覆盖度。",
        ),
    ]
    trace.verification = VerificationResult(
        accepted=False,
        support_rate=0.75,
        violations=["executor:missing_evidence"],
        next_action="escalate",
    )
    return trace


def test_run_trace_json_contains_required_fields(tmp_path: Path) -> None:
    run_trace = build_run_trace(_sample_trace(), run_id="run-test")
    output_path = save_run_trace(run_trace, runs_dir=tmp_path / "runs")
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["run_id"] == "run-test"
    assert data["task"]
    assert data["task_profile"]["complexity"] == 0.55
    assert data["task_profile"]["verifiability"] == 0.75
    assert data["selected_topology"] == "supervisor_worker"
    assert data["agents_executed"] == ["planner", "executor", "executor", "verifier"]
    assert len(data["contract_messages"]) == 4
    assert data["verification_result"]["accepted"] is False
    assert data["final_answer"]
    assert data["metrics"]["support_rate"] == 0.75


def test_contract_report_calculates_rates_and_violations() -> None:
    report = build_contract_report(_sample_trace())

    assert report.total_messages == 4
    assert report.supported_claims == 3
    assert report.support_rate == 0.75
    assert report.evidence_coverage == 0.75
    assert report.action_completeness == 1.0
    assert "Executor#3 missing evidence" in report.violations


def test_empty_evidence_lowers_support_rate() -> None:
    trace = _sample_trace()
    for message in trace.messages:
        message.evidence = []

    report = build_contract_report(trace)

    assert report.supported_claims == 0
    assert report.support_rate == 0.0
    assert report.evidence_coverage == 0.0


def test_contract_report_format_matches_human_readable_sections() -> None:
    rendered = format_contract_report(_sample_trace())

    assert "[Task Profile]" in rendered
    assert "complexity=0.55" in rendered
    assert "verifiability=0.75" in rendered
    assert "[Topology]" in rendered
    assert "selected=SUPERVISOR_WORKER" in rendered
    assert "[Contract Report]" in rendered
    assert "support_rate=0.75" in rendered
    assert "[Final Answer]" in rendered


def test_cli_observability_flags_do_not_break_default_output(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "总结可验证多智能体协同架构的核心目标。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert "答案" in result.stdout or "绛旀" in result.stdout
    assert "调试轨迹" in result.stdout or "璋冭瘯" in result.stdout


def test_cli_json_trace_outputs_parseable_json(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "总结可验证多智能体协同架构的核心目标。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--json-trace",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["run_id"]
    assert data["task"]
    assert data["contract_messages"]
    assert data["metrics"]["total_messages"] >= 1


def test_cli_save_run_writes_trace_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "总结可验证多智能体协同架构的核心目标。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--save-run",
            "--hide-trace",
        ],
    )

    assert result.exit_code == 0
    trace_files = list((tmp_path / "runs").glob("*/trace.json"))
    assert len(trace_files) == 1
    data = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert data["run_id"] == trace_files[0].parent.name
