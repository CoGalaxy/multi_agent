from typer.testing import CliRunner

import verifiable_multi_agent.cli as cli_module
from verifiable_multi_agent.cli import app

from tests.fake_llm import FakeLlmBackend


def _patch_backend(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "_build_backend", lambda *args, **kwargs: (FakeLlmBackend(), None))


def test_cli_shows_trace_by_default(tmp_path, monkeypatch) -> None:
    _patch_backend(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "总结可验证多智能体协同架构的核心目标。",
            "--backend",
            "vllm",
            "--memory",
            str(tmp_path / "memory.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert "Answer" in result.stdout
    assert "Topology" in result.stdout
    assert "Contract Trace" in result.stdout


def test_cli_can_hide_trace_when_requested(tmp_path, monkeypatch) -> None:
    _patch_backend(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "总结可验证多智能体协同架构的核心目标。",
            "--backend",
            "vllm",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--hide-trace",
        ],
    )

    assert result.exit_code == 0
    assert "Answer" in result.stdout
    assert "Topology" not in result.stdout
    assert "Contract Trace" not in result.stdout
