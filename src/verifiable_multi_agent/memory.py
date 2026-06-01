"""
协议记忆 — 存储和检索成功的协作模式。

与传统对话记忆不同：这里存的是"元层次"的协作结构（用了什么拓扑、
什么角色序列、什么 action），而不是对话内容本身。

设计意图：当接到与历史任务相似的新任务时，系统可以直接复用已验证的
协作模式（谁做什么、怎么验证），而不是每次都重新协商。

当前检索是简单的词重叠打分，后续替换为 embedding 相似度检索
以支持语义匹配——这又是一个"从规则到学习"的演进点。
"""

from __future__ import annotations

import json
from pathlib import Path

from verifiable_multi_agent.contracts import AgentTrace


class JsonlProtocolMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def store(self, trace: AgentTrace) -> None:
        """只存储通过验证的协作模式，未通过的不复用。"""
        if not trace.verification or not trace.verification.accepted:
            return
        record = {
            "task": trace.task,
            "topology": trace.topology.value,
            "support_rate": trace.verification.support_rate,
            "roles": [message.role.value for message in trace.messages],
            "actions": [message.action for message in trace.messages],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def retrieve(self, task: str, limit: int = 3) -> list[dict]:
        """按任务关键词重叠度排序，只返回有交集的记录。"""
        if not self.path.exists():
            return []
        task_terms = set(task.lower().split())
        scored: list[tuple[int, dict]] = []
        with self.path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                record = json.loads(line)
                score = len(task_terms & set(record["task"].lower().split()))
                scored.append((score, record))
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        return [record for score, record in ranked[:limit] if score > 0]
