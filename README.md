# Verifiable Multi-Agent Scaffold

> 面向复杂长程任务的可验证 LLM Agent 与多 Agent 协同架构研究

当前版本是论文方向的最小可运行基线。核心研究对象：

- **Contract Messages** — Agent 间结构化通信（claim + evidence + action）
- **Task Profiling** — 二维画像 (complexity, verifiability) 驱动拓扑选择
- **Adaptive Topology Routing** — SINGLE_AGENT / SUPERVISOR_WORKER / REVIEW_LOOP
- **Contract Verification** — 双层验证：结构合规 + 语义审计
- **Protocol Memory** — 路由决策记忆，基于历史数据修正拓扑选择
- **LLM Agent** — JSON 解析输出，与 RuleBasedAgent 接口一致

## Quick Start

```powershell
pip install -e ".[dev]"
vma "用三句话解释什么是二分查找。" --backend mock
pytest
```

## Backends

| 后端 | 说明 |
|------|------|
| `--backend mock` | 零依赖，RuleBasedAgent 确定性模板 |
| `--backend deepseek` | DeepSeek API，LLMAgent 驱动 |
| `--backend ollama` | 本地 Ollama，LLMAgent 驱动 |
| `--backend vllm` | 本地 vLLM，LLMAgent 驱动 |

```powershell
# Mock（无 LLM）
vma "Summarize this task." --backend mock

# DeepSeek
$env:DEEPSEEK_API_KEY="sk-..."
vma "比较 AutoGen 和 LangGraph 的架构差异" --backend deepseek --router quant

# Ollama 本地
vma "解释多 Agent 协同的优势" --backend ollama --model qwen3.5:4b
```

## Architecture

```
Task → Profiler (complexity, verifiability) → Router → Orchestrator → Agents → Verifier → Memory
```

**拓扑决策矩阵：**

| 条件 | 拓扑 |
|------|------|
| complexity < 0.4 且 verifiability < 0.4 | SINGLE_AGENT |
| complexity ≥ 0.4 且 verifiability < 0.4 | SUPERVISOR_WORKER |
| verifiability ≥ 0.4 | REVIEW_LOOP |

## Task Profiling

二维画像替代了旧的六维 TCI：

- **complexity** (0–1): 任务有多难分解（因果推理、多步规划、跨领域综合）
- **verifiability** (0–1): 输出有多难被验证（核查/证伪 ≥ 0.6，比较/分析 0.3–0.5，总结/定义 < 0.3）

评估方式：
- `--backend mock`：规则启发式（关键词 + 从句数）
- `--backend deepseek`：LLM 评估（few-shot prompt，JSON 输出）

## Router Modes

```powershell
# Legacy（默认）— 简单决策矩阵
vma "task" --router legacy

# Quant — 量化路由 + 动态拓扑图生成
vma "task" --router quant --show-topology --contract-report
```

`--router quant` 启用完整路径：`TaskProfile → TopologySpec → CollaborationGraph → GraphExecutor`

## Routing Memory

基于历史数据的拓扑修正。检索相似任务画像，若邻居失败率 > 0.5 则升级拓扑：

```
SINGLE_AGENT → SUPERVISOR_WORKER → REVIEW_LOOP
```

```powershell
# 默认启用，持久化到 runs/memory.json
vma "task" --router quant --routing-memory

# 禁用
vma "task" --router quant --no-routing-memory
```

## Contract Report

```powershell
vma "比较 AutoGen 和 LangGraph 的架构差异并验证比较标准。" \
    --backend mock --router quant --contract-report
```

输出示例：

```text
[Task Profile]
complexity=0.45 | verifiability=0.33

[Topology]
selected=SUPERVISOR_WORKER
reason=complexity=0.45, verifiability=0.33 → SUPERVISOR_WORKER

[Quantitative Router]
task_type=comparison
complexity=0.45 | verifiability=0.33
capability_needs=['planning', 'verification', 'synthesis']
generation_reasons=['complexity=0.45, verifiability=0.33', 'signal=comparison']

[Contract Report]
messages=4
support_rate=0.75
accepted=True

[Final Answer]
...
```

## Observable Runs

```powershell
# Contract report
vma "task" --backend mock --contract-report

# JSON trace
vma "task" --backend mock --json-trace

# 持久化到 runs/{run_id}/trace.json
vma "task" --backend mock --save-run
```

## Smoke Test

10 条中文任务覆盖 5 种类型，验证画像 → 路由 → 执行全流程：

```powershell
# 全部 10 条（DeepSeek，需要 DEEPSEEK_API_KEY）
python scripts/smoke_test.py

# 单条调试
python scripts/smoke_test.py -k "T09"

# 指定后端
python scripts/smoke_test.py --backend deepseek
```

结果写入 `runs/smoke_results.csv`，可直接复制进论文数据表。

## Test Suite

```powershell
pytest                          # 全部 72 项单元测试
pytest tests/test_memory.py     # ProtocolMemory 测试
pytest tests/test_llm_agent.py  # LLMAgent 集成测试
```
