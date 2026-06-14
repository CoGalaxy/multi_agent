from pathlib import Path

from verifiable_multi_agent.contracts import Topology
from verifiable_multi_agent.orchestrator import Orchestrator

from tests.fake_llm import FakeLlmBackend


def test_single_agent_quant_graph_has_executor_verifier_synthesizer(tmp_path: Path) -> None:
    trace = Orchestrator(tmp_path / "memory.jsonl", backend=FakeLlmBackend()).solve("Summarize this task.")

    assert trace.topology == Topology.SINGLE_AGENT
    assert trace.metadata["graph_execution"]["executed_nodes"] == ["executor", "verifier", "synthesizer"]
    assert trace.final_answer


def test_supervisor_worker_quant_graph_has_planner(tmp_path: Path) -> None:
    task = "Build a multi-agent coordination system with task allocation and communication protocols."
    trace = Orchestrator(tmp_path / "memory.jsonl", backend=FakeLlmBackend()).solve(task)

    assert trace.topology == Topology.SUPERVISOR_WORKER
    assert "planner" in trace.metadata["graph_execution"]["executed_nodes"]
    assert "executor" in trace.metadata["graph_execution"]["executed_nodes"]


def test_review_loop_quant_graph_has_safety_verifier(tmp_path: Path) -> None:
    task = "Investigate a high-risk security task, produce a safe plan, and verify compliance."
    trace = Orchestrator(tmp_path / "memory.jsonl", backend=FakeLlmBackend()).solve(task)

    assert trace.topology == Topology.REVIEW_LOOP
    assert "safety_verifier" in trace.metadata["graph_execution"]["executed_nodes"]
    assert "verifier" in trace.metadata["graph_execution"]["executed_nodes"]


def test_chinese_compare_task_keeps_quant_metadata(tmp_path: Path) -> None:
    task = "比较 AutoGen 和 CAMEL 作为多智能体协作基线的优缺点，并验证比较标准是否合理。"
    trace = Orchestrator(tmp_path / "memory.jsonl", backend=FakeLlmBackend()).solve(task)

    assert trace.metadata["router_mode"] == "quant"
    assert trace.metadata["topology_spec"]["capability_needs"]["verification"]
    assert trace.final_answer
