"""Main thesis orchestrator.

The runtime path is intentionally small and explicit:

Task text
-> TaskProfile
-> QuantitativeRouter
-> TopologySpec
-> CollaborationGraph
-> GraphExecutor
-> run trace and final answer

Only the quant route is kept because it is the thesis main line.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.constrained_topology import (
    ConstrainedTopologyGenerator,
    QuantitativeRouter,
    explain_topology_spec,
    infer_input_requirements,
    topology_from_spec,
)
from verifiable_multi_agent.contracts import AgentTrace, Budget, TaskProfile, Topology
from verifiable_multi_agent.memory import JsonlProtocolMemory
from verifiable_multi_agent.profiler import LlmProfiler, profile_task
from verifiable_multi_agent.rag import SimpleRagRetriever
from verifiable_multi_agent.routing_memory import MemoryRecord, ProtocolMemory
from verifiable_multi_agent.topology.executor import GraphExecutor


class Orchestrator:
    """Coordinates the quant-only multi-agent workflow."""

    def __init__(
        self,
        memory_path: Path | None = None,
        backend: LlmBackend | None = None,
        profiler: LlmProfiler | None = None,
        router_mode: str = "quant",
        rag_corpus_path: Path | None = None,
        routing_memory: ProtocolMemory | None = None,
    ) -> None:
        if router_mode != "quant":
            raise ValueError("Only quant router is kept in the thesis main line.")
        self.memory = JsonlProtocolMemory(memory_path or Path("data/protocol_memory.jsonl"))
        self.backend = backend
        self.profiler = profiler
        self.router_mode = router_mode
        self.retriever = SimpleRagRetriever(rag_corpus_path) if rag_corpus_path else None
        self._routing_memory = routing_memory

    def solve(self, task: str, budget: Budget | None = None) -> AgentTrace:
        """Run the thesis main workflow with a real LLM backend."""
        if not self.backend or not self.backend.is_real_llm:
            raise RuntimeError("A real LLM backend is required. Use vllm, deepseek, or ollama.")

        profile = self.profiler.profile(task) if self.profiler else profile_task(task)
        requirements = infer_input_requirements(task)
        topology_spec = QuantitativeRouter().route(
            profile,
            requirements,
            memory=self._routing_memory,
        )
        topology = topology_from_spec(topology_spec)
        topology_reason = explain_topology_spec(topology_spec)
        graph = ConstrainedTopologyGenerator().generate(topology_spec)

        trace = GraphExecutor(backend=self.backend, retriever=self.retriever).execute(
            task=task,
            graph=graph,
            profile=profile,
            topology=topology,
            topology_reason=topology_reason,
        )
        if _needs_verifier_rejection_upgrade(trace) and topology != Topology.REVIEW_LOOP:
            topology_spec = _upgrade_spec_for_verifier_rejection(topology_spec)
            topology = Topology.REVIEW_LOOP
            topology_reason = f"{explain_topology_spec(topology_spec)}; upgraded_after_verifier_rejected"
            graph = ConstrainedTopologyGenerator().generate(topology_spec)
            trace = GraphExecutor(backend=self.backend, retriever=self.retriever).execute(
                task=task,
                graph=graph,
                profile=profile,
                topology=topology,
                topology_reason=topology_reason,
            )
            trace.metadata["review_loop_upgrade_reason"] = "verifier_rejected"
        trace.metadata["router_mode"] = "quant"
        trace.metadata["task_profile"] = profile.model_dump(mode="json")
        trace.metadata["input_requirements"] = requirements.model_dump(mode="json")
        trace.metadata["topology_spec"] = topology_spec.model_dump(mode="json")

        self.memory.store(trace)
        self._add_routing_record(profile, trace)
        return trace

    def _add_routing_record(self, profile: TaskProfile, trace: AgentTrace) -> None:
        """Store the routing outcome for thesis topic 3."""
        if self._routing_memory is None:
            return
        self._routing_memory.add(
            MemoryRecord(
                task_id=str(uuid4()),
                complexity=profile.complexity,
                verifiability=profile.verifiability,
                topology_used=trace.topology.value,
                accepted=trace.verification.accepted if trace.verification else False,
                support_rate=trace.verification.support_rate if trace.verification else 0.0,
            )
        )


def _needs_verifier_rejection_upgrade(trace: AgentTrace) -> bool:
    if trace.verification is None or trace.verification.accepted:
        return False
    return any("verifier_rejected" in violation for violation in trace.verification.violations)


def _upgrade_spec_for_verifier_rejection(topology_spec):
    upgraded = topology_spec.model_copy(deep=True)
    upgraded.capability_needs.planning = True
    upgraded.capability_needs.verification = True
    upgraded.capability_needs.revision = True
    upgraded.capability_needs.synthesis = True
    upgraded.max_review_loops = max(upgraded.max_review_loops, 1)
    upgraded.max_nodes = max(upgraded.max_nodes, 5)
    upgraded.max_edges = max(upgraded.max_edges, upgraded.max_nodes - 1 + upgraded.max_review_loops)
    upgraded.generation_reasons = [*upgraded.generation_reasons, "upgrade=verifier_rejected_review_loop"]
    return upgraded
