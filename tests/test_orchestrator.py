from pathlib import Path

from verifiable_multi_agent.contracts import Topology
from verifiable_multi_agent.orchestrator import Orchestrator


def test_simple_task_uses_small_topology(tmp_path: Path) -> None:
    trace = Orchestrator(tmp_path / "memory.jsonl").solve("Summarize this task.")

    assert trace.topology == Topology.SINGLE_AGENT
    assert trace.verification is not None
    assert trace.verification.accepted
    assert trace.final_answer


def test_risky_task_uses_review_loop(tmp_path: Path) -> None:
    task = "Investigate a security task, verify evidence, then write a safe answer."
    trace = Orchestrator(tmp_path / "memory.jsonl").solve(task)

    assert trace.topology == Topology.REVIEW_LOOP
    # 高验证难度任务应该触发 REVIEW_LOOP
    assert trace.profile.verifiability >= 0.4
    assert trace.verification is not None
    assert trace.verification.accepted


def test_protocol_memory_is_written(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    Orchestrator(memory_path).solve("Compare evidence and verify the result.")

    assert memory_path.exists()
    assert memory_path.read_text(encoding="utf-8").strip()
