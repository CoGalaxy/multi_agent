"""
Agent 实现 — 规则驱动（mock）和 LLM 驱动两种模式。

RuleBasedAgent: 确定性模板（mock 模式），零依赖跑通管线。
LLMAgent:       调用 LLM 后端，要求 JSON 格式输出，解析后填充 ContractMessage。

接入本地模型时这里的改动最小：只需将 backend 替换为指向
http://127.0.0.1:8000/v1 的 OpenAICompatibleBackend 即可。
"""

from __future__ import annotations

import json
import logging

from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.contracts import AgentRole, ContractMessage
from verifiable_multi_agent.prompts import prompt_with_context, system_prompt

logger = logging.getLogger(__name__)


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
                    system_prompt("planning", task, "Return one concise plan claim."),
                    prompt_with_context(task, context, extra=stage_note),
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
                    system_prompt("execution", task, "Draft a concise candidate answer."),
                    prompt_with_context(
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
        # ── Verifier: 结构检查 + 语义审计 ──
        if self.role == AgentRole.VERIFIER:
            unsupported = [message.id for message in context if not message.has_support]
            if self.backend:
                # 让 LLM 做真正的语义验证，不只是结构化检查
                claim = self.backend.complete(
                    system_prompt(
                        "strict verification",
                        task,
                        "Audit the trace for topic drift, evidence quality, logical gaps, and task coverage. "
                        "Be specific about what is wrong. If everything checks out, say why.",
                    ),
                    prompt_with_context(task, context, extra=stage_note),
                    role=self.role.value,
                )
            else:
                claim = (
                    "All prior claims are supported and grounded in the task."
                    if not unsupported
                    else f"{len(unsupported)} message(s) lack evidence support."
                )
            return ContractMessage(
                role=self.role,
                subtask=subtask or "Verify contract trace — structural and semantic audit",
                claim=claim,
                evidence=[
                    f"Audited {len(context)} prior messages for claim-evidence alignment.",
                    f"Checked for topic drift, logical consistency, and task coverage.",
                ],
                action=action or ("Accept trace." if not unsupported else "Reject trace — escalate or retry."),
                uncertainty=0.15 if not unsupported else 0.65,
                metadata={
                    "unsupported_message_ids": unsupported,
                    "messages_audited": len(context),
                    **({"stage_note": stage_note} if stage_note else {}),
                },
            )
        # ── Synthesizer (默认分支): 综合所有消息产出最终答案 ──
        claim = _latest_claim(context, AgentRole.EXECUTOR) or task
        if self.backend:
            claim = self.backend.complete(
                system_prompt(
                    "synthesis",
                    task,
                    (
                        "Produce a concise final answer to the original task. "
                        "Use the internal trace to improve the answer, but do not mention internal orchestration, "
                        "topology, planner, executor, verifier, synthesizer, or trace unless the user asks. "
                        "Do not describe the internal route; if a procedure is needed, call it a plan, checklist, or steps."
                    ),
                ),
                prompt_with_context(task, context, extra=stage_note),
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


# ── LLM 驱动 Agent ────────────────────────────────────────────────


class LLMAgent:
    """LLM 驱动 Agent — 调用后端 LLM，要求 JSON 格式输出。

    与 RuleBasedAgent 接口一致，但始终使用 LLM 生成完整 ContractMessage，
    而非角色特定的确定性模板。解析失败时自动 fallback。

    适用于 --backend ollama / deepseek / vllm 等真实 LLM 后端。
    """

    def __init__(self, role: AgentRole, backend: LlmBackend) -> None:
        """初始化 LLM 驱动 Agent。

        Args:
            role: Agent 在协作流水线中的角色。
            backend: LLM 后端实例（必须提供，不支持 mock 模式）。
        """
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
        """调用 LLM 完成子任务，解析 JSON 输出为 ContractMessage。

        Args:
            task: 原始用户任务。
            context: 当前已有的消息上下文。
            subtask: 本步骤的子任务描述。
            action: 期望的执行动作（用于 fallback）。
            stage_note: 阶段提示（注入 user prompt）。

        Returns:
            解析成功的 ContractMessage，或解析失败时的 fallback 消息。
        """
        system = (
            f"你是一个 {self.role.value} Agent。只输出 JSON，不要任何额外内容。"
        )
        user = self._build_user_prompt(task, context, subtask, stage_note)
        try:
            raw = self.backend.complete(
                system=system, user=user, role=self.role.value
            )
            data = json.loads(raw)
            return ContractMessage(
                role=self.role,
                subtask=subtask or "",
                claim=str(data.get("claim", "")),
                evidence=_ensure_str_list(data.get("evidence", [])),
                action=str(data.get("action", action or "")),
                uncertainty=float(data.get("uncertainty", 0.5)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "LLMAgent parse failed for role=%s: %s", self.role.value, exc
            )
            return ContractMessage(
                role=self.role,
                subtask=subtask or "",
                claim="parse error",
                evidence=[],
                action=action or "",
                uncertainty=1.0,
            )

    # ── 内部方法 ─────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(
        task: str,
        context: list[ContractMessage],
        subtask: str | None,
        stage_note: str | None,
    ) -> str:
        """构造发送给 LLM 的 user prompt。"""
        context_summary = "\n".join(
            f"[{m.role.value}] {m.claim}" for m in context[-5:]
        )
        parts = [
            f"完成以下子任务：{subtask or '处理当前步骤'}",
            "",
            f"原始任务：{task}",
        ]
        if stage_note:
            parts.append(f"阶段提示：{stage_note}")
        if context_summary:
            parts.append(f"上下文消息（最近 5 条）：\n{context_summary}")
        parts.extend(
            [
                "",
                "输出格式（严格 JSON）：",
                "{",
                '  "claim": "核心主张，一句话",',
                '  "evidence": ["证据1", "证据2"],',
                '  "action": "analyze|compare|verify|synthesize 之一",',
                '  "uncertainty": 0.0到1.0的浮点数',
                "}",
            ]
        )
        return "\n".join(parts)


def _ensure_str_list(value: object) -> list[str]:
    """将输入转为字符串列表，过滤非字符串元素。"""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]
