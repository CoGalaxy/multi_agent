"""
任务画像模块 — 输入自然语言任务，输出多维特征向量。

提供两种实现：
- profile_task(): 关键词匹配规则（baseline），零延迟、可复现
- LlmProfiler: 用小模型（如 Qwen3.5:0.8b）做 few-shot 分类，
  准确率更高，作为论文中"从规则到学习"的演进对比。

实验设计：baseline (rules) vs 0.6B vs 4B 三组消融，分析分类器
参数量对整体协作质量的影响。
"""

from __future__ import annotations

import json
import logging
import re

from verifiable_multi_agent.contracts import TaskProfile

logger = logging.getLogger(__name__)

# 三类关键词表反映任务特征维度
TOOL_HINTS = {"search", "browse", "file", "api", "database", "tool", "evidence", "verify"}
RISK_HINTS = {"delete", "payment", "medical", "legal", "security", "harm", "private", "unsafe"}
UNCERTAINTY_HINTS = {"unknown", "maybe", "investigate", "compare", "latest", "current", "research"}

# few-shot prompt — 用三个典型样本教 0.8b 模型学会分类
_PROFILE_PROMPT = """Classify this task into four dimensions. Output ONLY a JSON object.

Dimension definitions:
- tool_need (0-1): Requires database, API, file system, web search, or external tools?
- uncertainty (0-1): Contains exploratory, ambiguous, or investigative language?
- step_count (int>=1): Number of distinct actions or clauses.
- risk (0-1): Involves deletion, payments, medical, legal, security, privacy, or irreversible operations?

Examples:
1. Say hello → {"tool_need": 0.0, "uncertainty": 0.0, "step_count": 1, "risk": 0.0}
2. Research latest AI trends and compare them → {"tool_need": 0.7, "uncertainty": 0.9, "step_count": 2, "risk": 0.0}
3. Delete user data from production DB, verify backup → {"tool_need": 1.0, "uncertainty": 0.1, "step_count": 2, "risk": 0.95}

Now classify: "{task}"

JSON:"""


def profile_task(task: str) -> TaskProfile:
    """关键词匹配 baseline — 零延迟，作为 LLM profiler 的 fallback。"""
    tokens = set(re.findall(r"[a-zA-Z_]+", task.lower()))
    clauses = re.split(r"[,;，；。]| and | then | after ", task.lower())
    step_count = max(1, len([clause for clause in clauses if clause.strip()]))

    return TaskProfile(
        task=task,
        tool_need=_score_overlap(tokens, TOOL_HINTS),
        uncertainty=_score_overlap(tokens, UNCERTAINTY_HINTS),
        step_count=step_count,
        risk=_score_overlap(tokens, RISK_HINTS),
    )


def _score_overlap(tokens: set[str], hints: set[str]) -> float:
    hits = len(tokens & hints)
    return min(hits / 2, 1.0)


class LlmProfiler:
    """用小模型（0.6B-1.5B）做任务画像，替代关键词规则。

    如果 LLM 返回的 JSON 解析失败，自动 fallback 到 profile_task()。
    """

    def __init__(self, backend) -> None:
        self._backend = backend

    def profile(self, task: str) -> TaskProfile:
        prompt = _PROFILE_PROMPT.replace("{task}", task)
        try:
            raw = self._backend.complete(system="", user=prompt)
            data = json.loads(raw)
            return TaskProfile(
                task=task,
                tool_need=float(data["tool_need"]),
                uncertainty=float(data["uncertainty"]),
                step_count=max(1, int(data["step_count"])),
                risk=float(data["risk"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("LLM profiler failed (%s), falling back to keyword rules", exc)
            return profile_task(task)
