# 合约消息有效性实验

本目录用于验证结构化合约消息在复杂任务中的作用。实验不主张多智能体直接提升模型生成质量，而关注合约消息是否能让模型显式给出证据，并提高任务覆盖检查、交付物缺失发现、事实幻觉暴露和失败可诊断能力。

## 实验分组

- `baseline`：直接调用同一 DeepSeek 模型回答任务，不要求结构化合约消息。
- `contract`：调用现有多智能体主流程，生成任务画像、量化路由、协作拓扑、合约消息、合约报告和最终答案。

## 运行方式

先确认项目根目录 `.env` 中存在 `DEEPSEEK_API_KEY`，然后在项目根目录运行：

```powershell
conda run -n vllm --no-capture-output python experiments/contract_message_effectiveness/run_experiment.py --limit 3
conda run -n vllm --no-capture-output python experiments/contract_message_effectiveness/evaluate_results.py
```

完整实验去掉 `--limit`：

```powershell
conda run -n vllm --no-capture-output python experiments/contract_message_effectiveness/run_experiment.py
conda run -n vllm --no-capture-output python experiments/contract_message_effectiveness/evaluate_results.py
```

## 输出文件

- `outputs/raw_runs.jsonl`：每个任务的 baseline 和 contract 原始运行结果。
- `outputs/evaluated_runs.jsonl`：加入幻觉评估、交付物评估后的逐条结果。
- `outputs/summary.json`：汇总指标。

## 主要指标

- `task_coverage`：系统合约报告中的任务覆盖率。
- `deliverable_complete`：最终答案是否实际包含任务要求的交付物。
- `fact_hallucination`：最终答案是否出现与给定事实冲突或无依据的事实。
- `deliverable_hallucination`：最终答案是否声称已完成某交付物但实际没有给出。
- `hallucination_detected_by_contract`：合约报告是否通过 `accepted=False` 或 `violations` 暴露幻觉风险。

## 实验解释

若实验结果显示 `contract` 组相较 `baseline` 组具有更低幻觉率，或在出现幻觉时具有更高检出率，则可说明结构化合约消息对复杂任务执行过程具有更强的可检查性与失败诊断能力。
