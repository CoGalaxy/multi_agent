from typer.testing import CliRunner

from verifiable_multi_agent.cli import app


def test_cli_shows_trace_by_default(tmp_path) -> None:
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
    assert "答案" in result.stdout
    assert "拓扑" in result.stdout
    assert "合约轨迹" in result.stdout


def test_cli_can_hide_trace_when_requested(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "总结可验证多智能体协同架构的核心目标。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--hide-trace",
        ],
    )

    assert result.exit_code == 0
    assert "答案" in result.stdout
    assert "拓扑" not in result.stdout
    assert "合约轨迹" not in result.stdout
