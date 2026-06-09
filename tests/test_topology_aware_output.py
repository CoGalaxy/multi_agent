from pathlib import Path

from verifiable_multi_agent.contracts import Topology
from verifiable_multi_agent.orchestrator import Orchestrator


def test_single_agent_trace_explains_direct_route(tmp_path: Path) -> None:
    trace = Orchestrator(tmp_path / "memory.jsonl").solve("Summarize this task.")

    # 新版 verifier 逐 evidence 评分可能触发 escalation
    assert trace.topology in (Topology.SINGLE_AGENT, Topology.REVIEW_LOOP)
    assert trace.topology_reason
    # 无论如何，subtask 中应该包含直接回答步骤（escalation 前执行）
    assert any("Directly answer" in message.subtask for message in trace.messages)
    assert trace.final_answer


def test_supervisor_worker_trace_explains_planner_handoff(tmp_path: Path) -> None:
    # 复杂度高但验证难度低的纯工程任务 → SUPERVISOR_WORKER
    task = "Build a multi-agent coordination system with task allocation and communication protocols."
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


def test_chinese_compare_task_uses_review_loop_and_chinese_trace(tmp_path: Path) -> None:
    # 比较+验证 → 高验证难度，路由到 REVIEW_LOOP
    task = "比较 AutoGen 和 CAMEL 作为多智能体协作基线的优缺点，并验证比较标准是否合理。"
    trace = Orchestrator(tmp_path / "memory.jsonl").solve(task)

    assert trace.topology == Topology.REVIEW_LOOP
    assert trace.topology_reason and "高验证难度" in trace.topology_reason
    assert any("初版安全回答" in message.subtask for message in trace.messages)
    assert trace.final_answer
    assert "执行路线" not in trace.final_answer


def test_chinese_high_risk_task_uses_review_loop(tmp_path: Path) -> None:
    task = "针对一个涉及安全约束的高风险工具调用任务，生成安全执行计划，并验证其合规性。"
    trace = Orchestrator(tmp_path / "memory.jsonl").solve(task)

    assert trace.topology == Topology.REVIEW_LOOP
    assert any("初版安全回答" in message.subtask for message in trace.messages)
    assert any("最终合规验证" in message.subtask for message in trace.messages)
