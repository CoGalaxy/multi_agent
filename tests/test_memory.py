from pathlib import Path

from verifiable_multi_agent.memory import JsonlProtocolMemory
from verifiable_multi_agent.contracts import AgentRole
from verifiable_multi_agent.orchestrator import Orchestrator


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


def test_protocol_memory_injects_memory_role_not_planner(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    memory_path.write_text(
        '{"task": "比较 AutoGen 和 CAMEL", "topology": "supervisor_worker", "support_rate": 1.0, "roles": [], "actions": []}\n',
        encoding="utf-8",
    )

    trace = Orchestrator(memory_path).solve("比较 AutoGen 和 CAMEL 的优缺点。")

    assert trace.messages[0].role == AgentRole.MEMORY
    assert trace.messages[0].subtask == "检索相似历史协作协议"
