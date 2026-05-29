# 可验证多智能体协作框架

本仓库是新毕设方向的首个可运行基线：

> 面向复杂长程任务的可验证 LLM Agent 与多 Agent 协同架构研究

当前版本刻意保持精简。它无需模型 API 即可运行，并将核心研究对象显式化：

- 智能体之间的合约消息
- 自适应拓扑路由
- Planner / Executor / Verifier 角色
- 合约级过程验证
- 用于记录可复用协作模式的协议记忆

## 快速开始

```powershell
python -m pip install -e ".[dev]"
vma "收集证据，给出简明答案，并附上验证说明。"
pytest
```

## DeepSeek 后端

在 shell 中设置 API 密钥：

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
vma "规划、执行并验证一项研究任务。" --backend deepseek --model deepseek-v4-flash --judge-model deepseek-v4-pro
```

`deepseek-v4-flash` 用于普通 agent 工作。当提供 `--judge-model` 时，`deepseek-v4-pro` 可保留给 verifier 和 synthesizer 角色使用。

## 架构

```mermaid
flowchart LR
    U[任务 Task] --> P[任务画像 Task Profiler]
    P --> R[拓扑路由 Topology Router]
    R --> O[编排器 Orchestrator]
    O --> A[智能体 Agents]
    A --> C[合约消息 Contract Messages]
    C --> V[合约验证器 Contract Verifier]
    V --> S[综合器 Synthesizer]
    S --> M[协议记忆 Protocol Memory]
```

### 三种运行拓扑

| 拓扑 | 触发条件 | Agent 执行序列 |
|---|---|---|
| SINGLE_AGENT | 低复杂度、低风险 | Executor → Verifier |
| SUPERVISOR_WORKER | 中等复杂度（≥0.25）或多步骤（≥3） | Planner → Executor → Verifier |
| REVIEW_LOOP | 高风险（≥0.5）或高复杂度（≥0.55） | Planner → Executor → Verifier → Executor → Verifier |

若首次验证未通过且当前拓扑低于 REVIEW_LOOP，编排器会自动升级为 REVIEW_LOOP 进行重试。

### 合约消息模型

Agent 之间的每次通信均为 `ContractMessage`，包含以下字段：

- `role` — Planner / Executor / Verifier / Synthesizer
- `subtask` — 当前子任务描述
- `claim` — Agent 的主张
- `evidence` — 支撑主张的证据列表
- `action` — 采取的行动
- `uncertainty` — 不确定性评分（0~1）
- `budget_hint` — 预算提示

验证器要求支持率 ≥ 80% 且无缺失 action，合约才算通过。

## 当前范围

本框架使用确定性 agent（`RuleBasedAgent`），以便在接入真实 LLM 后端之前完成基线日志和基准测试包装器的测试与稳定工作。

切换至 LLM 后端后，`RuleBasedAgent.run()` 会自动委托给后端，同时注入 grounding 规则——确保每次 LLM 调用都锚定原始任务，不引入任务之外的新话题。
