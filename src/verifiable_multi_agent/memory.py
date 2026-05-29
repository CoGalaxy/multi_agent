from __future__ import annotations

import json
from pathlib import Path

from verifiable_multi_agent.contracts import AgentTrace


class JsonlProtocolMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def store(self, trace: AgentTrace) -> None:
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
        if not self.path.exists():
            return []
        task_terms = set(task.lower().split())
        scored: list[tuple[int, dict]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                score = len(task_terms & set(record["task"].lower().split()))
                scored.append((score, record))
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        return [record for score, record in ranked[:limit] if score > 0]
