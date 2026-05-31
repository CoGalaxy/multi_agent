from pathlib import Path

from verifiable_multi_agent.contracts import Topology
from verifiable_multi_agent.orchestrator import Orchestrator


def test_single_agent_trace_explains_direct_route(tmp_path: Path) -> None:
    trace = Orchestrator(tmp_path / "memory.jsonl").solve("Summarize this task.")

    assert trace.topology == Topology.SINGLE_AGENT
    assert trace.topology_reason
    assert any("Directly answer" in message.subtask for message in trace.messages)
    assert any("Low complexity" in item for item in trace.execution_summary)


def test_supervisor_worker_trace_explains_planner_handoff(tmp_path: Path) -> None:
    task = "Compare AutoGen and CAMEL, then verify the comparison criteria."
    trace = Orchestrator(tmp_path / "memory.jsonl").solve(task)

    assert trace.topology == Topology.SUPERVISOR_WORKER
    assert any("Define work order" in message.subtask for message in trace.messages)
    assert any("Execute the planner" in message.subtask for message in trace.messages)
    assert any("planned criteria" in message.subtask for message in trace.messages)


def test_review_loop_trace_explains_revision_pass(tmp_path: Path) -> None:
    task = "Investigate a high-risk security task, produce a safe plan, and verify compliance."
    trace = Orchestrator(tmp_path / "memory.jsonl").solve(task)

    assert trace.topology == Topology.REVIEW_LOOP
    assert any("Draft initial safe answer" in message.subtask for message in trace.messages)
    assert any("Revise answer after verification" in message.subtask for message in trace.messages)
    assert any("Final compliance verification" in message.subtask for message in trace.messages)
