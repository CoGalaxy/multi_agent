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
from verifiable_multi_agent.contracts import AgentRole, AgentTrace, Budget, ContractMessage, Topology
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
                ContractMessage(
                    role=AgentRole.MEMORY,
                    subtask="检索相似历史协作协议" if _contains_cjk(task) else "Retrieve similar protocol memory",
                    claim=(
                        f"命中历史协作协议：{prior_protocols[0]['topology']}"
                        if _contains_cjk(task)
                        else f"Found reusable protocol: {prior_protocols[0]['topology']}"
                    ),
                    evidence=[f"memory_task={prior_protocols[0]['task']}"],
                    action="作为路由和执行提示，不直接承担规划角色。"
                    if _contains_cjk(task)
                    else "Use as routing and execution hint without impersonating the planner.",
                    metadata={"protocol_memory": prior_protocols[0]},
                )
            )
            budget.consume()

        # 阶段 1: 按拓扑执行 Agent 链
        for step in _steps_for(topology, task):
            budget.consume()
            trace.execution_summary.append(f"{_role_label(step['role'], task)}: {step['subtask']}")
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
        synth_stage_label = "执行摘要" if _contains_cjk(task) else "Execution summary"
        synth = RuleBasedAgent(AgentRole.SYNTHESIZER, backend=self.backend).run(
            task,
            trace.messages,
            subtask="合成最终答案" if _contains_cjk(task) else None,
            action="结合执行路线、候选答案和验证意见，返回最终答案。" if _contains_cjk(task) else None,
            stage_note=f"{synth_stage_label}:\n" + "\n".join(f"- {item}" for item in trace.execution_summary),
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


def _steps_for(topology: Topology, task: str = "") -> tuple[dict, ...]:
    if _contains_cjk(task):
        return _steps_for_zh(topology)
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


def _steps_for_zh(topology: Topology) -> tuple[dict, ...]:
    if topology == Topology.SINGLE_AGENT:
        return (
            {
                "role": AgentRole.EXECUTOR,
                "subtask": "直接回答低复杂度任务",
                "action": "不经过规划交接，直接给出简洁明确的回答。",
                "stage_note": "低复杂度路线：直接回答，但保留必要假设。",
            },
            {
                "role": AgentRole.VERIFIER,
                "subtask": "对直接回答进行轻量验证",
                "action": "检查回答是否紧扣原始任务，并给出明确结论。",
                "stage_note": "验证直接回答是否与原始任务一致，避免顾左右而言它。",
            },
        )
    if topology == Topology.SUPERVISOR_WORKER:
        return (
            {
                "role": AgentRole.PLANNER,
                "subtask": "定义工作顺序和比较标准",
                "action": "将任务拆成回答要求、比较维度和验证要点。",
                "stage_note": "中等复杂度路线：规划角色先明确回答结构和判断标准。",
            },
            {
                "role": AgentRole.EXECUTOR,
                "subtask": "按照规划结果执行回答",
                "action": "严格依据规划角色给出的标准生成答案。",
                "stage_note": "执行角色必须先给确定答复，再补充条件和限制。",
            },
            {
                "role": AgentRole.VERIFIER,
                "subtask": "按规划标准验证答案",
                "action": "检查答案是否覆盖每个标准，并指出证据或条件不足之处。",
                "stage_note": "验证角色按规划标准检查覆盖度，而不是泛泛确认。",
            },
        )
    return (
        {
            "role": AgentRole.PLANNER,
            "subtask": "规划高风险执行和合规检查",
            "action": "先识别安全边界、必要检查项和修订标准。",
            "stage_note": "高风险路线：在回答前先定义安全边界。",
        },
        {
            "role": AgentRole.EXECUTOR,
            "subtask": "生成初版安全回答",
            "action": "生成避免危险细节、说明缺失条件的初版答案。",
            "stage_note": "第一轮执行：给出安全初稿并暴露不确定性。",
        },
        {
            "role": AgentRole.VERIFIER,
            "subtask": "审查初稿的安全性和合规性",
            "action": "检查是否存在危险指令、约束缺失或无依据结论。",
            "stage_note": "第一轮验证：指出修订轮需要处理的问题。",
        },
        {
            "role": AgentRole.EXECUTOR,
            "subtask": "根据验证意见修订答案",
            "action": "根据验证意见修订答案，并明确安全理由。",
            "stage_note": "第二轮执行：根据验证意见修订初稿。",
        },
        {
            "role": AgentRole.VERIFIER,
            "subtask": "最终合规验证",
            "action": "确认修订后的答案满足安全和任务约束。",
            "stage_note": "最终验证：只有安全且紧扣任务时才接受。",
        },
    )


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _role_label(role: AgentRole, task: str) -> str:
    if not _contains_cjk(task):
        return role.value
    return {
        AgentRole.PLANNER: "规划角色",
        AgentRole.EXECUTOR: "执行角色",
        AgentRole.VERIFIER: "验证角色",
        AgentRole.SYNTHESIZER: "合成角色",
        AgentRole.MEMORY: "记忆角色",
    }[role]
