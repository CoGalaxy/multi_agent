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

# 关键词表 — 中英双语，反映任务特征维度
TOOL_HINTS = {
    "search", "browse", "file", "api", "database", "tool", "evidence", "verify",
    "搜索", "浏览", "文件", "数据库", "工具", "证据", "验证", "查询",
    "下载", "上传", "接口", "部署", "配置", "安装", "监控", "爬虫",
}
RISK_HINTS = {
    "delete", "payment", "medical", "legal", "security", "harm", "private", "unsafe",
    "删除", "支付", "医疗", "法律", "安全", "隐私", "危险", "密码",
    "审计", "合规", "加密", "认证", "权限", "审批", "财务", "生产",
}
UNCERTAINTY_HINTS = {
    "unknown", "maybe", "investigate", "compare", "latest", "current", "research",
    "未知", "可能", "调查", "比较", "最新", "当前", "研究", "探索",
    "分析", "评估", "优化", "设计", "规划", "建议", "也许", "尝试",
}

# few-shot prompt — 中英双语样例教模型分类（Qwen 中文原生，双语 prompt 效果最佳）
_PROFILE_PROMPT = """Classify this task into four dimensions. Output ONLY a JSON object.
Classify both Chinese and English tasks equally well.

Dimensions:
- tool_need (0-1): Requires database, API, file system, web search, or external tools? (需要数据库、API、文件系统、搜索等工具?)
- uncertainty (0-1): Contains exploratory, ambiguous, or investigative language? (包含探索性、模糊、调查性表述?)
- step_count (int>=1): Number of distinct actions or clauses. (独立步骤或从句数量)
- risk (0-1): Involves deletion, payments, medical, legal, security, privacy, or irreversible ops? (涉及删除、支付、医疗、法律、安全、隐私等高风险?)

Examples:
1. Say hello → {"tool_need": 0.0, "uncertainty": 0.0, "step_count": 1, "risk": 0.0}
2. 打个招呼 → {"tool_need": 0.0, "uncertainty": 0.0, "step_count": 1, "risk": 0.0}
3. Research latest AI trends and compare them → {"tool_need": 0.7, "uncertainty": 0.9, "step_count": 2, "risk": 0.0}
4. 研究最新AI趋势并做对比分析 → {"tool_need": 0.7, "uncertainty": 0.9, "step_count": 2, "risk": 0.0}
5. Delete user data from production DB, verify backup → {"tool_need": 1.0, "uncertainty": 0.1, "step_count": 2, "risk": 0.95}
6. 删除生产数据库中的用户数据并验证备份 → {"tool_need": 1.0, "uncertainty": 0.1, "step_count": 2, "risk": 0.95}
7. 设计一个复杂的多智能体协作系统，包括任务分配、通信协议和冲突解决 → {"tool_need": 0.4, "uncertainty": 0.75, "step_count": 3, "risk": 0.1}

Now classify: "{task}"

JSON:"""


def profile_task(task: str) -> TaskProfile:
    """关键词匹配 baseline — 零延迟，中英双语支持。

    中文用子串匹配（无空格分词的折中方案），英文用 token set 匹配。
    """
    text = task.lower()

    # 英文 token set 匹配
    en_tokens = set(re.findall(r"[a-zA-Z_]+", text))
    # 中文子串匹配：遍历关键词表看是否出现在原文中
    cn_hits = {h for h in TOOL_HINTS | RISK_HINTS | UNCERTAINTY_HINTS if h in text}

    # 按中英文标点 + 连接词拆分从句
    clauses = re.split(r"[,;，；。、\n]| and | then | after | 然后 | 之后 | 以及 | 并且 ", text)
    step_count = max(1, len([c for c in clauses if c.strip()]))
    all_hints = en_tokens | cn_hits

    return TaskProfile(
        task=task,
        tool_need=_score_overlap(all_hints, TOOL_HINTS),
        uncertainty=_score_overlap(all_hints, UNCERTAINTY_HINTS),
        step_count=step_count,
        risk=_score_overlap(all_hints, RISK_HINTS),
    )


def _score_overlap(hits: set[str], hints: set[str]) -> float:
    n = len(hits & hints)
    return min(n / 2, 1.0)


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
