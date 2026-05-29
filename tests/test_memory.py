from pathlib import Path

from verifiable_multi_agent.memory import JsonlProtocolMemory


def test_memory_retrieve_handles_tied_scores(tmp_path: Path) -> None:
    memory = JsonlProtocolMemory(tmp_path / "memory.jsonl")
    memory.path.write_text(
        "\n".join(
            [
                '{"task": "compare agent baselines", "topology": "single_agent", "support_rate": 1.0, "roles": [], "actions": []}',
                '{"task": "compare agent methods", "topology": "review_loop", "support_rate": 1.0, "roles": [], "actions": []}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    results = memory.retrieve("compare agent", limit=2)

    assert len(results) == 2
