from __future__ import annotations

from verifiable_multi_agent.agents import LLMAgent, RuleBasedAgent
from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.contracts import AgentRole, AgentTrace, ContractMessage, TaskProfile, Topology, VerificationResult
from verifiable_multi_agent.rag import SimpleRagRetriever
from verifiable_multi_agent.topology.graph import AgentNode, AgentType, CollaborationGraph
from verifiable_multi_agent.topology.validator import validate_graph
from verifiable_multi_agent.verifier import verify_contracts


class GraphExecutor:
    def __init__(self, backend: LlmBackend | None = None, retriever: SimpleRagRetriever | None = None) -> None:
        self.backend = backend
        self.retriever = retriever

    def execute(
        self,
        task: str,
        graph: CollaborationGraph,
        profile: TaskProfile,
        topology: Topology,
        topology_reason: str,
    ) -> AgentTrace:
        trace = AgentTrace(
            task=task,
            topology=topology,
            profile=profile,
            topology_reason=topology_reason,
            execution_summary=[topology_reason],
            metadata={
                "generated_topology": graph.model_dump(mode="json"),
                "graph_execution": {
                    "enabled": True,
                    "executed_nodes": [],
                    "skipped_nodes": [],
                    "review_loops_used": 0,
                    "execution_mode": "sequential_dag",
                },
            },
        )

        if graph.blocked:
            trace.verification = VerificationResult(
                accepted=False,
                support_rate=0.0,
                violations=[graph.block_reason or "blocked"],
                next_action="blocked",
            )
            trace.final_answer = graph.block_reason
            return trace

        validation_errors = validate_graph(graph)
        graph.validation_errors = validation_errors
        trace.metadata["generated_topology"] = graph.model_dump(mode="json")
        if validation_errors:
            trace.verification = VerificationResult(
                accepted=False,
                support_rate=0.0,
                violations=validation_errors,
                next_action="blocked",
            )
            trace.final_answer = "Generated topology failed validation."
            trace.metadata["graph_execution"]["skipped_nodes"] = graph.node_types()
            return trace

        for node in graph.nodes:
            message = self._run_node(task, node, trace.messages)
            trace.messages.append(message)
            trace.execution_summary.append(f"{node.type.value}: {node.label or node.type.value}")
            trace.metadata["graph_execution"]["executed_nodes"].append(node.type.value)

        trace.verification = verify_contracts(trace.messages)
        trace = self._maybe_run_review_loop(task, graph, trace)
        trace.final_answer = _final_answer(trace.messages) or task
        return trace

    def _run_node(self, task: str, node: AgentNode, context: list[ContractMessage]) -> ContractMessage:
        role = _role_for_node(node.type)
        stage_note = f"Graph node={node.type.value}; config={node.config}"
        if node.type == AgentType.RESEARCHER and self.retriever and self.retriever.available:
            hits = self.retriever.search(task)
            if hits:
                evidence = [hit.evidence_line() for hit in hits]
                stage_note = (
                    f"{stage_note}\nRetrieved RAG evidence:\n"
                    + "\n".join(f"- {line}" for line in evidence)
                )
                message = self._make_agent(role).run(
                    task,
                    context,
                    subtask=_subtask_for_node(node.type),
                    action="Retrieve top-k RAG passages and ground downstream execution in them.",
                    stage_note=stage_note,
                )
                message.evidence = evidence
                message.metadata.update(
                    {
                        "node_id": node.id,
                        "node_type": node.type.value,
                        "node_config": node.config,
                        "rag_hits": [
                            {
                                "id": hit.document.id,
                                "title": hit.document.title,
                                "source": hit.document.source,
                                "score": hit.score,
                            }
                            for hit in hits
                        ],
                    }
                )
                return message
        message = self._make_agent(role).run(
            task,
            context,
            subtask=_subtask_for_node(node.type),
            action=_action_for_node(node.type),
            stage_note=stage_note,
        )
        message.metadata.update({"node_id": node.id, "node_type": node.type.value, "node_config": node.config})
        if node.type == AgentType.TOOL_EXECUTOR:
            message.evidence.append("mock_tool_result: no real tool was invoked in stage 2.")
        return message

    def _maybe_run_review_loop(self, task: str, graph: CollaborationGraph, trace: AgentTrace) -> AgentTrace:
        if trace.verification is None or trace.verification.accepted:
            return trace
        if graph.constraints.max_review_loops <= 0:
            return trace
        if not any(node.type == AgentType.REVISER for node in graph.nodes):
            return trace

        loops_used = 0
        while loops_used < graph.constraints.max_review_loops and trace.verification and not trace.verification.accepted:
            reviser = AgentNode(id=f"reviser_loop_{loops_used + 1}", type=AgentType.REVISER, label="Reviser")
            verifier = AgentNode(id=f"verifier_loop_{loops_used + 1}", type=AgentType.VERIFIER, label="Verifier")
            for node in (reviser, verifier):
                trace.messages.append(self._run_node(task, node, trace.messages))
                trace.metadata["graph_execution"]["executed_nodes"].append(node.type.value)
            loops_used += 1
            trace.metadata["graph_execution"]["review_loops_used"] = loops_used
            trace.verification = verify_contracts(trace.messages)
        return trace

    def _make_agent(self, role: AgentRole) -> RuleBasedAgent | LLMAgent:
        """根据 backend 类型选择合适的 Agent 实现。"""
        if self.backend and self.backend.is_real_llm:
            return LLMAgent(role, self.backend)
        return RuleBasedAgent(role, backend=self.backend)


def _role_for_node(node_type: AgentType) -> AgentRole:
    if node_type == AgentType.PLANNER:
        return AgentRole.PLANNER
    if node_type in {AgentType.VERIFIER, AgentType.SAFETY_VERIFIER, AgentType.TESTER}:
        return AgentRole.VERIFIER
    if node_type == AgentType.SYNTHESIZER:
        return AgentRole.SYNTHESIZER
    return AgentRole.EXECUTOR


def _subtask_for_node(node_type: AgentType) -> str:
    return {
        AgentType.PLANNER: "Plan graph execution steps",
        AgentType.RESEARCHER: "Extract material-grounded evidence",
        AgentType.TOOL_EXECUTOR: "Run required tool step",
        AgentType.EXECUTOR: "Produce candidate solution",
        AgentType.CODER: "Implement candidate code",
        AgentType.TESTER: "Test candidate code",
        AgentType.CRITIC: "Critique candidate solution",
        AgentType.REVISER: "Revise candidate solution",
        AgentType.SAFETY_VERIFIER: "Review safety constraints",
        AgentType.VERIFIER: "Verify graph execution output",
        AgentType.SYNTHESIZER: "Synthesize final answer",
    }[node_type]


def _action_for_node(node_type: AgentType) -> str:
    return {
        AgentType.PLANNER: "Create a graph-local execution plan.",
        AgentType.RESEARCHER: "Ground the answer in provided material or note missing material.",
        AgentType.TOOL_EXECUTOR: "Produce a mock tool observation for deterministic testing.",
        AgentType.EXECUTOR: "Draft the answer using upstream messages.",
        AgentType.CODER: "Draft implementation-oriented output.",
        AgentType.TESTER: "Check the candidate output with deterministic test reasoning.",
        AgentType.CRITIC: "Identify gaps before revision.",
        AgentType.REVISER: "Revise using critique or verifier concerns.",
        AgentType.SAFETY_VERIFIER: "Check safety and policy-sensitive constraints.",
        AgentType.VERIFIER: "Verify support, coverage, and consistency.",
        AgentType.SYNTHESIZER: "Return the final response from graph messages.",
    }[node_type]


def _final_answer(messages: list[ContractMessage]) -> str | None:
    for message in reversed(messages):
        if message.metadata.get("node_type") == AgentType.SYNTHESIZER.value:
            return message.claim
    return messages[-1].claim if messages else None
