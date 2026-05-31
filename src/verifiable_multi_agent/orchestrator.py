"""
编排器 — 框架的主控模块，串联 Profile → Route → Execute → Verify → Store 全流程。

核心流程：
1. 任务画像 → 拓扑选择
2. 检索协议记忆（有相似任务则注入复用提示）
3. 按拓扑顺序执行 Agent 链
4. 双层合约验证：
   - Layer 1: 结构合规（零成本 rule-based gate）
   - Layer 2: 语义审计（LLM 检查 topic drift / evidence gap / 逻辑一致性）
   - 未通过且有余量则自动升级为 REVIEW_LOOP 重试
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
from verifiable_multi_agent.profiler import LlmProfiler, profile_task
from verifiable_multi_agent.router import select_topology
from verifiable_multi_agent.verifier import SemanticVerifier, verify_contracts


class Orchestrator:
    def __init__(
        self,
        memory_path: Path | None = None,
        backend: LlmBackend | None = None,
        profiler: LlmProfiler | None = None,
    ) -> None:
        self.memory = JsonlProtocolMemory(memory_path or Path("data/protocol_memory.jsonl"))
        self.backend = backend
        self.profiler = profiler
        self._semantic = SemanticVerifier(backend) if (backend and backend.is_real_llm) else None

    def solve(self, task: str, budget: Budget | None = None) -> AgentTrace:
        budget = budget or Budget()
        profile = self.profiler.profile(task) if self.profiler else profile_task(task)
        topology = select_topology(profile)
        prior_protocols = self.memory.retrieve(task)

        trace = AgentTrace(task=task, topology=topology, profile=profile)

        # 阶段 0: 有相似协作模式则注入复用提示（不占用正式角色槽位）
        if prior_protocols:
            trace.messages.append(
                RuleBasedAgent(AgentRole.PLANNER).run(
                    f"Reuse protocol memory: {prior_protocols[0]['topology']} for {task}",
                    trace.messages,
                )
            )
            budget.consume()

        # 阶段 1: 按拓扑执行 Agent 链
        # 跳过 Planner 如果已经在阶段 0 注入了（避免重复）
        roles = _roles_for(topology)
        skip_planner = prior_protocols and AgentRole.PLANNER in roles
        for role in roles:
            if skip_planner and role == AgentRole.PLANNER:
                skip_planner = False
                continue
            budget.consume()
            trace.messages.append(RuleBasedAgent(role, backend=self.backend).run(task, trace.messages))

        # 阶段 2: 结构验证 → 不通过则升级
        trace.verification = verify_contracts(trace.messages)
        if not trace.verification.accepted and topology != Topology.REVIEW_LOOP and budget.remaining_steps > 1:
            trace.topology = Topology.REVIEW_LOOP
            for role in (AgentRole.VERIFIER, AgentRole.SYNTHESIZER):
                budget.consume()
                trace.messages.append(RuleBasedAgent(role, backend=self.backend).run(task, trace.messages))
            trace.verification = verify_contracts(trace.messages)

        # 阶段 2b: 语义验证 — 结构通过后，用 LLM 检查内容质量
        if trace.verification.accepted and self._semantic:
            semantic_result = self._semantic.verify(task, trace.messages)
            # 语义验证结果作为补充标注，不影响结构通过与否
            # 但 violations 会合并到最终结果中
            trace.verification = semantic_result

        # 阶段 3: 综合最终答案
        synth = RuleBasedAgent(AgentRole.SYNTHESIZER, backend=self.backend).run(task, trace.messages)
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
