import json

from typer.testing import CliRunner

from verifiable_multi_agent.cli import app
from verifiable_multi_agent.contracts import TaskProfile, Topology
from verifiable_multi_agent.routing_spec import CapabilityNeeds, TaskType, TopologySpec
from verifiable_multi_agent.topology.constraints import TopologyConstraints
from verifiable_multi_agent.topology.executor import GraphExecutor
from verifiable_multi_agent.topology.generator import ConstrainedTopologyGenerator
from verifiable_multi_agent.topology.graph import AgentEdge, AgentNode, AgentType, CollaborationGraph
from verifiable_multi_agent.topology.validator import validate_graph


def _profile(task: str = "task") -> TaskProfile:
    return TaskProfile(task=task, complexity=0.4, verifiability=0.4)


def _spec(needs: CapabilityNeeds, max_nodes: int = 8, blocked: bool = False) -> TopologySpec:
    return TopologySpec(
        task_type=TaskType.GENERAL,
        complexity=0.4,
        verifiability=0.4,
        capability_needs=needs,
        max_nodes=max_nodes,
        max_edges=max(max_nodes - 1, 0),
        max_review_loops=1 if needs.safety_review else 0,
        max_tool_calls=3 if needs.tool_execution else 0,
        blocked=blocked,
        block_reason="missing material" if blocked else None,
        generation_reasons=["unit-test"],
    )


def _types(graph: CollaborationGraph) -> list[str]:
    return graph.node_types()


def test_simple_graph_executes_executor_verifier_synthesizer() -> None:
    graph = ConstrainedTopologyGenerator().generate(_spec(CapabilityNeeds(), max_nodes=3))
    trace = GraphExecutor().execute("用三句话解释什么是二分查找。", graph, _profile(), Topology.SINGLE_AGENT, "test")

    assert _types(graph) == ["executor", "verifier", "synthesizer"]
    assert trace.metadata["graph_execution"]["executed_nodes"] == ["executor", "verifier", "synthesizer"]


def test_comparison_graph_executes_planner_executor_verifier_synthesizer() -> None:
    graph = ConstrainedTopologyGenerator().generate(
        _spec(CapabilityNeeds(planning=True, verification=True, synthesis=True), max_nodes=4)
    )
    trace = GraphExecutor().execute("比较 AutoGen 和 CAMEL 的架构差异，并给出适用场景。", graph, _profile(), Topology.SUPERVISOR_WORKER, "test")

    assert _types(graph) == ["planner", "executor", "verifier", "synthesizer"]
    assert trace.metadata["graph_execution"]["executed_nodes"] == ["planner", "executor", "verifier", "synthesizer"]
    assert trace.final_answer
    assert len(trace.messages) >= 3


def test_material_grounding_graph_includes_researcher() -> None:
    graph = ConstrainedTopologyGenerator().generate(
        _spec(CapabilityNeeds(planning=True, material_grounding=True, verification=True, synthesis=True), max_nodes=5)
    )
    trace = GraphExecutor().execute("根据材料比较两个框架。材料如下：A 强，B 简单。", graph, _profile(), Topology.SUPERVISOR_WORKER, "test")

    assert _types(graph) == ["planner", "researcher", "executor", "verifier", "synthesizer"]
    assert "researcher" in trace.metadata["graph_execution"]["executed_nodes"]


def test_tool_execution_graph_includes_mock_tool_executor_message() -> None:
    graph = ConstrainedTopologyGenerator().generate(
        _spec(CapabilityNeeds(planning=True, tool_execution=True, verification=True, synthesis=True), max_nodes=5)
    )
    trace = GraphExecutor().execute("查询数据库中退款失败的订单，并说明原因。", graph, _profile(), Topology.SUPERVISOR_WORKER, "test")

    assert _types(graph) == ["planner", "tool_executor", "executor", "verifier", "synthesizer"]
    assert "tool_executor" in trace.metadata["graph_execution"]["executed_nodes"]
    assert any(message.metadata.get("node_type") == "tool_executor" for message in trace.messages)


def test_material_and_tool_graph_augments_evidence_template() -> None:
    graph = ConstrainedTopologyGenerator().generate(
        _spec(
            CapabilityNeeds(
                planning=True,
                material_grounding=True,
                tool_execution=True,
                verification=True,
                synthesis=True,
            ),
            max_nodes=6,
        )
    )

    assert _types(graph) == ["planner", "researcher", "tool_executor", "executor", "verifier", "synthesizer"]
    assert "base_template=evidence_grounded_template" in graph.generation_reasons
    assert "augment=insert_tool_executor_before_executor" in graph.generation_reasons


def test_comparison_critique_revision_augments_comparison_template() -> None:
    spec = TopologySpec(
        task_type=TaskType.COMPARISON,
        complexity=0.5,
        verifiability=0.6,
        capability_needs=CapabilityNeeds(
            planning=True,
            verification=True,
            synthesis=True,
            critique=True,
            revision=True,
        ),
        max_nodes=6,
        max_edges=5,
        max_review_loops=0,
        max_tool_calls=0,
        generation_reasons=["unit-test"],
    )

    graph = ConstrainedTopologyGenerator().generate(spec)

    assert _types(graph) == ["planner", "executor", "critic", "reviser", "verifier", "synthesizer"]
    assert "base_template=comparison_template" in graph.generation_reasons
    assert "augment=insert_critic_after_executor" in graph.generation_reasons
    assert "augment=insert_reviser_after_critic" in graph.generation_reasons


def test_code_testing_budget_failure_reports_validation_errors() -> None:
    graph = ConstrainedTopologyGenerator().generate(
        _spec(CapabilityNeeds(planning=True, code_testing=True, verification=True, synthesis=True), max_nodes=4)
    )
    trace = GraphExecutor().execute("实现并测试一个函数。", graph, _profile(), Topology.SUPERVISOR_WORKER, "test")

    assert "too_many_nodes:6>4" in graph.validation_errors
    assert trace.verification is not None
    assert not trace.verification.accepted
    assert "too_many_nodes:6>4" in trace.verification.violations


def test_safety_review_graph_includes_safety_verifier() -> None:
    graph = ConstrainedTopologyGenerator().generate(
        _spec(CapabilityNeeds(planning=True, safety_review=True, verification=True, synthesis=True), max_nodes=5)
    )
    trace = GraphExecutor().execute("删除生产数据库中的测试用户。", graph, _profile(), Topology.REVIEW_LOOP, "test")

    assert _types(graph) == ["planner", "executor", "safety_verifier", "verifier", "synthesizer"]
    assert "safety_verifier" in trace.metadata["graph_execution"]["executed_nodes"]


def test_blocked_graph_executes_no_regular_nodes() -> None:
    graph = ConstrainedTopologyGenerator().generate(
        _spec(CapabilityNeeds(planning=True, material_grounding=True, verification=True), max_nodes=5, blocked=True)
    )
    trace = GraphExecutor().execute("请根据给定材料比较。", graph, _profile(), Topology.SINGLE_AGENT, "blocked")

    assert graph.blocked
    assert trace.metadata["graph_execution"]["executed_nodes"] == []
    assert trace.verification is not None
    assert not trace.verification.accepted
    assert trace.final_answer


def test_validate_graph_finds_required_errors() -> None:
    constraints = TopologyConstraints(max_nodes=1, max_edges=0, max_review_loops=0, max_tool_calls=0)
    graph = CollaborationGraph(
        nodes=[AgentNode(id="executor", type=AgentType.EXECUTOR), AgentNode(id="extra", type=AgentType.CRITIC)],
        edges=[AgentEdge(source="executor", target="extra")],
        entry_node="missing",
        exit_node="extra",
        constraints=constraints,
    )

    errors = validate_graph(graph)

    assert "missing_verifier" in errors
    assert "missing_synthesizer" in errors
    assert "too_many_nodes:2>1" in errors
    assert "invalid_entry_node" in errors


def test_cli_quant_show_topology_contract_report(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "比较 AutoGen 和 CAMEL 的架构差异，并给出适用场景。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--router",
            "quant",
            "--show-topology",
            "--contract-report",
        ],
    )

    assert result.exit_code == 0
    assert "[Generated Topology]" in result.stdout
    assert "Planner → Executor → Verifier → Synthesizer" in result.stdout
    assert "[Graph Execution]" in result.stdout
    assert "executed_nodes" in result.stdout
    assert "[Contract Report]" in result.stdout


def test_json_trace_contains_generated_topology_and_graph_execution(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "比较 AutoGen 和 CAMEL 的架构差异，并给出适用场景。",
            "--backend",
            "mock",
            "--memory",
            str(tmp_path / "memory.jsonl"),
            "--router",
            "quant",
            "--json-trace",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["generated_topology"]["nodes"]
    assert data["graph_execution"]["enabled"]
    assert "planner" in data["graph_execution"]["executed_nodes"]
