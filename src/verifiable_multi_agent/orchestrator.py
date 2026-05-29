from __future__ import annotations

from pathlib import Path

from verifiable_multi_agent.agents import RuleBasedAgent
from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.contracts import AgentRole, AgentTrace, Budget, Topology
from verifiable_multi_agent.memory import JsonlProtocolMemory
from verifiable_multi_agent.profiler import profile_task
from verifiable_multi_agent.router import select_topology
from verifiable_multi_agent.verifier import verify_contracts


class Orchestrator:
    def __init__(self, memory_path: Path | None = None, backend: LlmBackend | None = None) -> None:
        self.memory = JsonlProtocolMemory(memory_path or Path("data/protocol_memory.jsonl"))
        self.backend = backend

    def solve(self, task: str, budget: Budget | None = None) -> AgentTrace:
        budget = budget or Budget()
        profile = profile_task(task)
        topology = select_topology(profile)
        prior_protocols = self.memory.retrieve(task)

        trace = AgentTrace(task=task, topology=topology, profile=profile)
        if prior_protocols:
            trace.messages.append(
                RuleBasedAgent(AgentRole.PLANNER).run(
                    f"Reuse protocol memory: {prior_protocols[0]['topology']} for {task}",
                    trace.messages,
                )
            )
            budget.consume()

        for role in _roles_for(topology):
            budget.consume()
            trace.messages.append(RuleBasedAgent(role, backend=self.backend).run(task, trace.messages))

        trace.verification = verify_contracts(trace.messages)
        if not trace.verification.accepted and topology != Topology.REVIEW_LOOP and budget.remaining_steps > 1:
            trace.topology = Topology.REVIEW_LOOP
            for role in (AgentRole.VERIFIER, AgentRole.SYNTHESIZER):
                budget.consume()
                trace.messages.append(RuleBasedAgent(role, backend=self.backend).run(task, trace.messages))
            trace.verification = verify_contracts(trace.messages)

        synth = RuleBasedAgent(AgentRole.SYNTHESIZER, backend=self.backend).run(task, trace.messages)
        trace.messages.append(synth)
        trace.final_answer = synth.claim
        self.memory.store(trace)
        return trace


def _roles_for(topology: Topology) -> tuple[AgentRole, ...]:
    if topology == Topology.SINGLE_AGENT:
        return (AgentRole.EXECUTOR, AgentRole.VERIFIER)
    if topology == Topology.SUPERVISOR_WORKER:
        return (AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.VERIFIER)
    return (AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.VERIFIER, AgentRole.EXECUTOR, AgentRole.VERIFIER)
