# Verifiable Multi-Agent Scaffold

面向复杂长程任务的可验证多智能体协同架构研究 — 最小可运行基线。

## 核心模块

| 模块 | 说明 |
|------|------|
| **Contract Messages** | Agent 间结构化通信：claim（主张）+ evidence（证据）+ action（动作） |
| **Task Profiling** | 二维画像：complexity（分解难度）+ verifiability（验证难度），由规则或 LLM 评估 |
| **Topology Routing** | 三种协作拓扑：SINGLE_AGENT / SUPERVISOR_WORKER / REVIEW_LOOP |
| **Contract Verifier** | 双层验证：结构合规（rule-based gate）+ 语义审计（LLM judge） |
| **Protocol Memory** | 路由决策记忆：基于历史相似任务画像修正拓扑选择 |
| **LLM Agent** | JSON 解析输出，与规则 Agent 接口一致，按 backend 自动切换 |

## Quick Start

```powershell
pip install -e ".[dev]"
vma "用三句话解释什么是二分查找。" --backend mock
pytest
```

## Backends

| 参数 | Agent | 说明 |
|------|-------|------|
| `--backend mock` | RuleBasedAgent | 零依赖，确定性模板 |
| `--backend ollama` | LLMAgent | 本地 Ollama 推理 |
| `--backend vllm` | LLMAgent | 本地 vLLM 推理 |
| `--backend deepseek` | LLMAgent | DeepSeek API 云端推理 |

```powershell
# Mock（无 LLM 依赖，快速验证管线）
vma "Summarize this task." --backend mock

# DeepSeek
$env:DEEPSEEK_API_KEY="sk-..."
vma "比较 AutoGen 和 LangGraph 的架构差异" --backend deepseek --router quant

# 本地 Ollama
vma "解释多 Agent 协同的优势" --backend ollama --model qwen3.5:4b
```

## Task Profiling

每个任务被评估为两个独立维度：

- **complexity**（0–1）：任务有多难分解。因果推理、多步规划、跨领域综合 → 高；一句话总结 → 低
- **verifiability**（0–1）：输出有多难被验证。核查/证伪 → 高（≥ 0.6）；比较/分析 → 中（0.3–0.5）；总结/定义 → 低（< 0.3）

`--backend mock` 时使用关键词启发式规则，`--backend deepseek` 时由 flash 模型评估。

## Topology Routing

决策矩阵（阈值 0.4）：

| complexity | verifiability | 拓扑 |
|-----------|--------------|------|
| < 0.4 | < 0.4 | SINGLE_AGENT — 直接执行 + 轻量验证 |
| ≥ 0.4 | < 0.4 | SUPERVISOR_WORKER — 规划 → 执行 → 验证 |
| 任意 | ≥ 0.4 | REVIEW_LOOP — 执行 → 审查 → 修订闭环 |

```powershell
# Legacy 路由（默认）
vma "task" --router legacy

# 量化路由 + 动态拓扑图
vma "task" --router quant --show-topology
```

`--router quant` 启用 `TopologySpec → CollaborationGraph → GraphExecutor` 完整路径。

## Routing Memory

检索历史相似任务的画像，若邻居失败率超过 50% 且当前拓扑非最高级，自动升级：

```
SINGLE_AGENT → SUPERVISOR_WORKER → REVIEW_LOOP
```

持久化到 `runs/memory.json`，默认启用。`--no-routing-memory` 可关闭。

## Contract Report

```powershell
vma "比较 AutoGen 和 LangGraph 的架构差异并验证比较标准。" \
    --backend mock --router quant --contract-report
```

```text
[Task Profile]
complexity=0.45 | verifiability=0.33

[Topology]
selected=SUPERVISOR_WORKER
reason=complexity=0.45, verifiability=0.33 → SUPERVISOR_WORKER

[Contract Report]
messages=4
support_rate=0.75
accepted=True

[Final Answer]
...
```

## 可观测运行

```powershell
vma "task" --backend mock --contract-report   # 合约报告
vma "task" --backend mock --json-trace        # JSON 轨迹
vma "task" --backend mock --save-run          # 持久化到 runs/{run_id}/trace.json
```

## Smoke Test

10 条中文任务覆盖事实比较、多跳推理、开放分析、验证挑战、路由边界五类：

```powershell
python scripts/smoke_test.py                  # 全部（需要 DEEPSEEK_API_KEY）
python scripts/smoke_test.py -k "T09"         # 单条调试
python scripts/smoke_test.py --backend all    # 多后端对比
```

每条任务输出 CSV 行，汇总写入 `runs/smoke_results.csv`。

## Tests

```powershell
pytest                          # 全部 72 项
pytest tests/test_memory.py     # 路由记忆
pytest tests/test_llm_agent.py  # LLM Agent 集成
```
