"""
协议记忆 — 存储和检索成功的协作模式。

包含两个互补的记忆系统：
- JsonlProtocolMemory: 存储"元层次"的协作结构（拓扑、角色序列、action），
  用于相似任务的模式复用。
- ProtocolMemory: 存储路由决策记录（任务画像 → 拓扑 → 结果），
  用于基于历史数据优化拓扑选择。

设计意图：当接到与历史任务相似的新任务时，系统可以直接复用已验证的
协作模式（谁做什么、怎么验证），而不是每次都重新协商。

当前检索是简单的词重叠打分，后续替换为 embedding 相似度检索
以支持语义匹配——这又是一个"从规则到学习"的演进点。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
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


# ── 路由决策记忆 ──────────────────────────────────────────────────


@dataclass
class MemoryRecord:
    """单条路由决策记录 — 记录一次任务画像→拓扑选择→验证结果。

    用于 ProtocolMemory 的存储与检索，支持基于历史数据优化拓扑选择。
    """

    task_id: str
    complexity: float
    verifiability: float
    topology_used: str
    accepted: bool
    support_rate: float


class ProtocolMemory:
    """路由决策记忆 — 基于历史画像相似度推荐拓扑。

    存储每次路由决策的结果（任务画像 + 选中的拓扑 + 验证是否通过），
    在新任务到来时检索最相似的历史记录，辅助拓扑选择决策。

    持久化格式为 JSON 数组，使用 dataclasses.asdict 序列化。
    """

    def __init__(self, path: str = "runs/memory.json") -> None:
        """初始化协议记忆。

        Args:
            path: JSON 持久化文件路径。若文件已存在则加载，否则初始化为空列表。
        """
        self._path = Path(path)
        self._records: list[MemoryRecord] = []
        if self._path.exists():
            self._load()

    # ── 公开方法 ─────────────────────────────────────────────────

    def add(self, record: MemoryRecord) -> None:
        """追加一条路由决策记录并立即持久化。

        Args:
            record: 路由决策记录（任务画像 + 拓扑选择 + 验证结果）。
        """
        self._records.append(record)
        self._persist()

    def query(
        self, complexity: float, verifiability: float, k: int = 3
    ) -> list[MemoryRecord]:
        """用欧氏距离在 (complexity, verifiability) 空间中找最近 k 条记录。

        Args:
            complexity: 查询任务的复杂度（0.0~1.0）。
            verifiability: 查询任务的可验证性（0.0~1.0）。
            k: 返回的最近邻数量，默认 3。

        Returns:
            按距离升序排列的最近 k 条记录。records 为空时返回空列表。
        """
        if not self._records:
            return []
        scored = [
            (
                math.sqrt(
                    (r.complexity - complexity) ** 2
                    + (r.verifiability - verifiability) ** 2
                ),
                r,
            )
            for r in self._records
        ]
        scored.sort(key=lambda item: item[0])
        return [r for _, r in scored[:k]]

    def historical_fail_rate(self, neighbors: list[MemoryRecord]) -> float:
        """计算邻居记录中 accepted=False 的比例。

        Args:
            neighbors: query() 返回的邻居记录列表。

        Returns:
            失败比例（0.0~1.0）。空列表返回 0.0。
        """
        if not neighbors:
            return 0.0
        failed = sum(1 for r in neighbors if not r.accepted)
        return round(failed / len(neighbors), 3)

    # ── 内部方法 ─────────────────────────────────────────────────

    def _persist(self) -> None:
        """将所有记录序列化为 JSON 数组写入文件。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(record) for record in self._records]
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        """从 JSON 文件加载记录列表。"""
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._records = [MemoryRecord(**item) for item in raw]
