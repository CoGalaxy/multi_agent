"""
编排器 — 框架的主控模块，串联 Profile → Route → Execute → Verify → Store 全流程。

核心流程：
1. 任务画像 → 拓扑选择
2. 检索协议记忆（有相似任务则注入复用提示）
3. 按拓扑顺序执行 Agent 链
4. 合约验证 → 未通过且有余量则自动升级为 REVIEW_LOOP 重试
5. Synthesizer 产出最终答案
6. 通过的协作模式存入协议记忆

自适应升级（escalation）是区别于固定管线框架的关键：系统在检测到
验证失败时自主选择更强的拓扑，而不是让用户手动调整。
"""

from __future__ import annotations

from pathlib import Path

from verifiable_multi_agent.agents import RuleBasedAgent
from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.contracts import AgentRole, AgentTrace, Budget, Topology
from verifiable_multi_agent.memory import JsonlProtocolMemory
from verifiable_multi_agent.profiler import profile_task
from verifiable_multi_agent.router import explain_topology, select_topology
from verifiable_multi_agent.verifier import verify_contracts


class Orchestrator:
    def __init__(self, memory_path: Path | None = None, backend: LlmBackend | None = None) -> None:
        self.memory = JsonlProtocolMemory(memory_path or Path("data/protocol_memory.jsonl"))
        self.backend = backend

    def solve(self, task: str, budget: Budget | None = None) -> AgentTrace:
        budget = budget or Budget()
        profile = profile_task(task)
        topology = select_topology(profile)
        topology_reason = explain_topology(profile, topology)
        prior_protocols = self.memory.retrieve(task)

        trace = AgentTrace(
            task=task,
            topology=topology,
            profile=profile,
            topology_reason=topology_reason,
            execution_summary=[topology_reason],
        )
        # 阶段 0: 如果有相似任务的协作模式，先注入复用提示
        if prior_protocols:
            trace.messages.append(
                RuleBasedAgent(AgentRole.PLANNER).run(
                    f"Reuse protocol memory: {prior_protocols[0]['topology']} for {task}",
                    trace.messages,
                )
            )
            budget.consume()

        # 阶段 1: 按拓扑执行 Agent 链
        for step in _steps_for(topology):
            budget.consume()
            trace.execution_summary.append(f"{step['role'].value}: {step['subtask']}")
            trace.messages.append(
                RuleBasedAgent(step["role"], backend=self.backend).run(
                    task,
                    trace.messages,
                    subtask=step["subtask"],
                    action=step["action"],
                    stage_note=step["stage_note"],
                )
            )

        # 阶段 2: 验证 → 不通过则升级拓扑重试
        trace.verification = verify_contracts(trace.messages)
        if not trace.verification.accepted and topology != Topology.REVIEW_LOOP and budget.remaining_steps > 1:
            # 自适应升级：从 SINGLE_AGENT 或 SUPERVISOR_WORKER 升级到 REVIEW_LOOP
            trace.topology = Topology.REVIEW_LOOP
            escalation_note = "Escalated after failed verification: perform a stronger review before synthesis."
            trace.topology_reason = escalation_note
            trace.execution_summary.append(escalation_note)
            for role in (AgentRole.VERIFIER, AgentRole.SYNTHESIZER):
                budget.consume()
                trace.messages.append(
                    RuleBasedAgent(role, backend=self.backend).run(
                        task,
                        trace.messages,
                        stage_note=escalation_note,
                    )
                )
            trace.verification = verify_contracts(trace.messages)

        # 阶段 3: 综合最终答案
        synth = RuleBasedAgent(AgentRole.SYNTHESIZER, backend=self.backend).run(
            task,
            trace.messages,
            stage_note="Execution summary:\n" + "\n".join(f"- {item}" for item in trace.execution_summary),
        )
        trace.messages.append(synth)
        trace.final_answer = synth.claim
        # 阶段 4: 通过的协作模式存入协议记忆
        self.memory.store(trace)
        return trace


def _roles_for(topology: Topology) -> tuple[AgentRole, ...]:
    """每种拓扑对应的 Agent 执行序列。REVIEW_LOOP 多一轮 Executor→Verifier。"""
    if topology == Topology.SINGLE_AGENT:
        return (AgentRole.EXECUTOR, AgentRole.VERIFIER)
    if topology == Topology.SUPERVISOR_WORKER:
        return (AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.VERIFIER)
    return (AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.VERIFIER, AgentRole.EXECUTOR, AgentRole.VERIFIER)


def _steps_for(topology: Topology) -> tuple[dict, ...]:
    if topology == Topology.SINGLE_AGENT:
        return (
            {
                "role": AgentRole.EXECUTOR,
                "subtask": "Directly answer the low-complexity task",
                "action": "Produce a concise direct answer without a planning handoff.",
                "stage_note": "Low-complexity route: answer directly, but keep assumptions explicit.",
            },
            {
                "role": AgentRole.VERIFIER,
                "subtask": "Lightweight verification of direct answer",
                "action": "Check that the direct answer remains on task and supported by the task statement.",
                "stage_note": "Verify whether the direct answer is grounded in the original task.",
            },
        )
    if topology == Topology.SUPERVISOR_WORKER:
        return (
            {
                "role": AgentRole.PLANNER,
                "subtask": "Define work order and comparison criteria",
                "action": "Break the task into criteria, answer requirements, and verification points.",
                "stage_note": "Medium-complexity route: planner defines the dimensions the executor should cover.",
            },
            {
                "role": AgentRole.EXECUTOR,
                "subtask": "Execute the planner's work order",
                "action": "Draft the answer by following the planner's criteria.",
                "stage_note": "Use the planner's criteria as the structure of the answer.",
            },
            {
                "role": AgentRole.VERIFIER,
                "subtask": "Verify answer against planned criteria",
                "action": "Check whether the answer covers each planned criterion and flags missing evidence.",
                "stage_note": "Verify coverage against the planner's criteria, not just generic support.",
            },
        )
    return (
        {
            "role": AgentRole.PLANNER,
            "subtask": "Plan high-risk execution and compliance checks",
            "action": "Identify safe boundaries, required checks, and revision criteria.",
            "stage_note": "High-risk route: define safety boundaries before any answer is drafted.",
        },
        {
            "role": AgentRole.EXECUTOR,
            "subtask": "Draft initial safe answer",
            "action": "Produce an initial answer that avoids unsafe specifics and states missing details.",
            "stage_note": "First executor pass: produce a safe draft and expose uncertainty.",
        },
        {
            "role": AgentRole.VERIFIER,
            "subtask": "Review safety and compliance of initial draft",
            "action": "Check for unsafe instructions, missing constraints, and unsupported claims.",
            "stage_note": "First verifier pass: identify issues the revision pass should address.",
        },
        {
            "role": AgentRole.EXECUTOR,
            "subtask": "Revise answer after verification",
            "action": "Revise the answer using verifier concerns and make the safety rationale explicit.",
            "stage_note": "Second executor pass: revise the draft according to verifier concerns.",
        },
        {
            "role": AgentRole.VERIFIER,
            "subtask": "Final compliance verification",
            "action": "Confirm the revised answer satisfies safety and task constraints.",
            "stage_note": "Final verifier pass: accept only if the revised answer is safe and on task.",
        },
    )
