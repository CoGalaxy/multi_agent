"""
任务画像模块 — 输入自然语言任务，输出多维特征向量。

当前版本用关键词匹配做轻量级特征提取，优势是零延迟、零成本、
可复现，适合中期前验证管线逻辑。

后续替换方向：用 7B 量级的本地模型做分类（tool_need/uncertainty/risk
分别训练三个二分类器，或直接让模型输出 JSON），在保持低成本的同时
提升准确率——这本身就是论文中可以讨论的设计取舍。
"""

from __future__ import annotations

import re

from verifiable_multi_agent.contracts import TaskProfile


# 三类关键词表反映任务特征维度。英文走 token 匹配，中文走子串匹配。
TOOL_HINTS = {"search", "browse", "file", "api", "database", "tool", "evidence", "verify"}
RISK_HINTS = {"delete", "payment", "medical", "legal", "security", "harm", "private", "unsafe"}
UNCERTAINTY_HINTS = {"unknown", "maybe", "investigate", "compare", "latest", "current", "research"}
TOOL_HINTS_ZH = {"工具", "证据", "验证", "检索", "搜索", "文件", "数据库", "调用"}
RISK_HINTS_ZH = {"高风险", "安全", "合规", "医疗", "法律", "隐私", "删除", "危险", "越权"}
UNCERTAINTY_HINTS_ZH = {"比较", "研究", "分析", "调查", "最新", "当前", "不确定", "评估"}


def profile_task(task: str) -> TaskProfile:
    """从自然语言任务中提取四个维度的特征分数。"""
    tokens = set(re.findall(r"[a-zA-Z_]+", task.lower()))
    # 按标点和连接词拆分从句，用于估计步骤数
    clauses = re.split(r"[,;，；。]| and | then | after ", task.lower())
    step_count = max(1, len([clause for clause in clauses if clause.strip()]))

    return TaskProfile(
        task=task,
        tool_need=_score_overlap(tokens, TOOL_HINTS, task, TOOL_HINTS_ZH),
        uncertainty=_score_overlap(tokens, UNCERTAINTY_HINTS, task, UNCERTAINTY_HINTS_ZH),
        step_count=step_count,
        risk=_score_overlap(tokens, RISK_HINTS, task, RISK_HINTS_ZH),
    )


def _score_overlap(tokens: set[str], hints: set[str], task: str, zh_hints: set[str]) -> float:
    """命中 2 个关键词即视为饱和 (score = 1.0)，避免少数命中就高估。"""
    hits = len(tokens & hints) + sum(1 for hint in zh_hints if hint in task)
    return min(hits / 2, 1.0)
