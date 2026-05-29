from __future__ import annotations

import re

from verifiable_multi_agent.contracts import TaskProfile


TOOL_HINTS = {"search", "browse", "file", "api", "database", "tool", "evidence", "verify"}
RISK_HINTS = {"delete", "payment", "medical", "legal", "security", "harm", "private", "unsafe"}
UNCERTAINTY_HINTS = {"unknown", "maybe", "investigate", "compare", "latest", "current", "research"}


def profile_task(task: str) -> TaskProfile:
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
