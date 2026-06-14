from verifiable_multi_agent.orchestrator import Orchestrator

from tests.fake_llm import FakeLlmBackend


def test_llm_calls_are_grounded_in_original_task(tmp_path) -> None:
    backend = FakeLlmBackend()
    task = "Plan, execute, and verify a research task."

    Orchestrator(tmp_path / "memory.jsonl", backend=backend).solve(task)

    assert backend.calls
    assert all(f"Original task: {task}" in call["user"] for call in backend.calls)
    assert any(call["role"] == "synthesizer" for call in backend.calls)


def test_chinese_task_is_sent_to_llm_backend(tmp_path) -> None:
    backend = FakeLlmBackend()
    task = "总结可验证多智能体协同架构的核心目标。"

    Orchestrator(tmp_path / "memory.jsonl", backend=backend).solve(task)

    assert backend.calls
    assert all(task in call["user"] for call in backend.calls)


def test_synthesizer_is_part_of_quant_graph(tmp_path) -> None:
    backend = FakeLlmBackend()
    task = "针对一个涉及安全约束的高风险工具调用任务，生成安全执行计划，并验证其合规性。"

    trace = Orchestrator(tmp_path / "memory.jsonl", backend=backend).solve(task)

    assert trace.metadata["router_mode"] == "quant"
    assert "synthesizer" in trace.metadata["graph_execution"]["executed_nodes"]
