"""
规则驱动 Agent — 中期前验证阶段的确定性实现。

每个 Agent 角色有一套默认的 deterministic 输出模板：当 backend 为
None 时直接返回模板（mock 模式），无需 LLM 即可跑通完整管线。

当 backend 不为 None 时，自动委托给 LLM 生成 claim，但在 system
prompt 中注入 GROUNDING_RULES 确保 LLM 不偏离原始任务。

接入本地模型时这里的改动最小：只需将 backend 替换为指向
http://127.0.0.1:8000/v1 的 OpenAICompatibleBackend 即可。
"""

from __future__ import annotations

from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.contracts import AgentRole, ContractMessage


# 所有 LLM 调用的 system prompt 前缀，防止模型发散
GROUNDING_RULES = (
    "Ground every claim in the original task. "
    "Do not introduce a new topic that is absent from the task. "
    "If the task is underspecified, say what is missing and provide a safe generic plan."
)


class RuleBasedAgent:
    def __init__(self, role: AgentRole, backend: LlmBackend | None = None) -> None:
        self.role = role
        self.backend = backend

    def run(
        self,
        task: str,
        context: list[ContractMessage],
        subtask: str | None = None,
        action: str | None = None,
        stage_note: str | None = None,
    ) -> ContractMessage:
        # ── Planner: 分解任务，产出执行计划 ──
        if self.role == AgentRole.PLANNER:
            claim = f"The task can be handled with {max(1, min(3, len(task) // 60 + 1))} focused step(s)."
            if self.backend:
                claim = self.backend.complete(
                    f"You are a planning agent. Return one concise plan claim. {GROUNDING_RULES}",
                    _prompt_with_context(task, context, extra=stage_note),
                    role=self.role.value,
                )
            return ContractMessage(
                role=self.role,
                subtask=subtask or "Decompose task and choose execution plan",
                claim=claim,
                evidence=["Task profile and requested deliverable were inspected."],
                action=action or "Create an execution checklist and hand it to the executor.",
                uncertainty=0.25,
                metadata={"stage_note": stage_note} if stage_note else {},
            )
        # ── Executor: 基于 Planner 的计划产出候选答案 ──
        if self.role == AgentRole.EXECUTOR:
            plan = _latest_claim(context, AgentRole.PLANNER)
            claim = f"Candidate answer for task: {task}"
            if self.backend:
                claim = self.backend.complete(
                    f"You are an execution agent. Draft a concise candidate answer. {GROUNDING_RULES}",
                    _prompt_with_context(
                        task,
                        context,
                        extra="\n".join(
                            item
                            for item in [stage_note, f"Selected plan: {plan or 'direct execution'}"]
                            if item
                        ),
                    ),
                    role=self.role.value,
                )
            return ContractMessage(
                role=self.role,
                subtask=subtask or "Produce candidate solution",
                claim=claim,
                evidence=[plan or "Direct task statement is available."],
                action=action or "Draft the answer and expose assumptions for verification.",
                uncertainty=0.35,
                metadata={"stage_note": stage_note} if stage_note else {},
            )
        # ── Verifier: 检查所有先序消息是否有支撑 ──
        if self.role == AgentRole.VERIFIER:
            unsupported = [message.id for message in context if not message.has_support]
            claim = "All prior claims are supported." if not unsupported else "Some claims lack support."
            if self.backend:
                claim = self.backend.complete(
                    f"You are a verification agent. Judge whether the trace is supported and on task. {GROUNDING_RULES}",
                    _prompt_with_context(task, context, extra=stage_note),
                    role=self.role.value,
                )
            return ContractMessage(
                role=self.role,
                subtask=subtask or "Check contract support",
                claim=claim,
                evidence=["Checked each contract message for claim and evidence fields."],
                action=action or ("Accept trace." if not unsupported else "Request local retry for unsupported messages."),
                uncertainty=0.15 if not unsupported else 0.65,
                metadata={
                    "unsupported_message_ids": unsupported,
                    **({"stage_note": stage_note} if stage_note else {}),
                },
            )
        # ── Synthesizer (默认分支): 综合所有消息产出最终答案 ──
        claim = _latest_claim(context, AgentRole.EXECUTOR) or task
        if stage_note:
            claim = f"Execution route:\n{stage_note}\n\nFinal answer:\n{claim}"
        if self.backend:
            claim = self.backend.complete(
                (
                    "You are a synthesis agent. Produce a concise final answer to the original task. "
                    "Start with a brief 'Execution route' sentence explaining how the selected topology shaped the answer. "
                    f"{GROUNDING_RULES}"
                ),
                _prompt_with_context(task, context, extra=stage_note),
                role=self.role.value,
            )
        return ContractMessage(
            role=self.role,
            subtask=subtask or "Synthesize final answer",
            claim=claim,
            evidence=[message.claim for message in context if message.has_support],
            action=action or "Return concise final response with trace summary.",
            uncertainty=0.2,
            metadata={"stage_note": stage_note} if stage_note else {},
        )


def _latest_claim(messages: list[ContractMessage], role: AgentRole) -> str | None:
    """从消息列表中取最近一条指定角色的 claim，供下游 Agent 引用。"""
    for message in reversed(messages):
        if message.role == role:
            return message.claim
    return None


def _prompt_with_context(task: str, context: list[ContractMessage], extra: str | None = None) -> str:
    """组装 LLM 输入：原始任务 + 完整协作轨迹 + 可选的计划提示。"""
    lines = [
        f"Original task: {task}",
        "Trace:",
    ]
    if not context:
        lines.append("- No prior messages.")
    for message in context:
        lines.append(
            f"- role={message.role.value}; subtask={message.subtask}; "
            f"claim={message.claim}; evidence={message.evidence}; action={message.action}"
        )
    if extra:
        lines.append(extra)
    return "\n".join(lines)
