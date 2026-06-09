"""
合约数据模型 — 框架的"类型骨架"。

所有 Agent 间通信、验证结果、运行轨迹都通过这些 Pydantic 模型
进行结构化约束。设计目标是让每一步协作都可追溯、可审计。

后续接入本地模型时，LLM 输出需要被解析为这些结构——这部分是最容易
出幻觉的地方，需要重点处理输出格式约束。
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """协作流水线中的角色。"""

    MEMORY = "memory"
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"


class Topology(str, Enum):
    """
    三种协作拓扑，按复杂度/风险递增。

    SINGLE_AGENT:    适用简单任务，跳过规划直接执行+验证
    SUPERVISOR_WORKER: 中等任务，先规划再执行
    REVIEW_LOOP:      高风险任务，执行-验证闭环迭代
    """

    SINGLE_AGENT = "single_agent"
    SUPERVISOR_WORKER = "supervisor_worker"
    REVIEW_LOOP = "review_loop"


class TaskProfile(BaseModel):
    """
    任务画像 — Profiler 的输出，Router 的输入。

    两个独立维度捕捉任务特征：
    - complexity:    任务有多难分解（0.0~1.0），由 LLM/规则对任务文本直接评估
    - verifiability: 输出有多难被验证（0.0~1.0），由 LLM/规则对任务文本直接评估

    拓扑决策矩阵：
      complexity < 0.4 且 verifiability < 0.4  → SINGLE_AGENT
      complexity >= 0.4 且 verifiability < 0.4 → SUPERVISOR_WORKER
      其他（verifiability >= 0.4）             → REVIEW_LOOP
    """

    task: str
    complexity: float = Field(ge=0.0, le=1.0, default=0.0)
    verifiability: float = Field(ge=0.0, le=1.0, default=0.0)


class Budget(BaseModel):
    """
    步数预算 — 防止协作进入无限循环。

    在接入本地模型后尤其重要：小模型可能反复 self-correct 但不收敛，
    预算耗尽时强制终止并返回当前最佳结果。
    """

    max_steps: int = 8
    max_messages: int = 12
    remaining_steps: int = 8

    def consume(self) -> None:
        if self.remaining_steps <= 0:
            raise RuntimeError("Budget exhausted")
        self.remaining_steps -= 1


class ContractMessage(BaseModel):
    """
    Agent 间通信的基本单元 — 框架的核心数据结构。

    每个消息必须声明：
    - claim:   Agent 的主张（我要做什么/我得出了什么）
    - evidence: 支撑主张的依据（我参考了什么）
    - action:  基于主张采取的行动（我实际做了什么）

    这三者构成一个"可检查的协作步骤"：验证器检查 claim 是否有 evidence
    支撑、action 是否非空、uncertainty 是否在合理范围。

    设计类比：软件工程中 PR 的 description + test evidence + CI check。
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: AgentRole
    subtask: str
    claim: str
    evidence: list[str] = Field(default_factory=list)
    action: str
    uncertainty: float = Field(ge=0.0, le=1.0, default=0.5)
    budget_hint: int = Field(ge=0, default=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_support(self) -> bool:
        """claim 同时有 evidence 和非空文本时才算有支撑。"""
        return bool(self.evidence) and self.claim.strip() != ""


class VerificationResult(BaseModel):
    """合约验证结果 — 不是最终答案的"对错"，而是协作过程是否合规。"""

    accepted: bool
    support_rate: float
    violations: list[str] = Field(default_factory=list)
    next_action: str = "accept"


class AgentTrace(BaseModel):
    """
    一次完整运行的全程记录。

    所有数据都存于此结构：输入任务、选中的拓扑、每一步的合约消息、
    验证结果、最终答案。Protocol Memory 从中提取可复用的协作模式存储。
    """

    task: str
    topology: Topology
    profile: TaskProfile
    topology_reason: str | None = None
    execution_summary: list[str] = Field(default_factory=list)
    messages: list[ContractMessage] = Field(default_factory=list)
    verification: VerificationResult | None = None
    final_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
