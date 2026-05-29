import os
from pathlib import Path

from verifiable_multi_agent.cli import load_env_file


def test_load_env_file_supports_env_reference(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-test")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=${ANTHROPIC_AUTH_TOKEN}\n", encoding="utf-8")

    load_env_file(env_file)

    assert os.environ["DEEPSEEK_API_KEY"] == "sk-test"
