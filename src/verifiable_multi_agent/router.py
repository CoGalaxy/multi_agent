"""
自适应拓扑路由 — 根据任务画像选择协作拓扑。

决策逻辑是一个两层的 if-else 树，阈值的设定逻辑是：
- risk >= 0.5：任务触及高风险关键词，必须走 REVIEW_LOOP
- complexity >= 0.55：综合复杂度偏高，同样需要闭环审查
- complexity >= 0.25 或 step_count >= 3：中等任务，加 Planner
- 其余：简单任务，省掉规划开销直接用 SINGLE_AGENT

这些阈值目前是经验值，后续在 benchmark 上做网格搜索调优
可以作为实验的一部分（超参数敏感性分析）。
"""

from __future__ import annotations

from verifiable_multi_agent.contracts import TaskProfile, Topology


def select_topology(profile: TaskProfile) -> Topology:
    if profile.risk >= 0.5 or profile.complexity >= 0.55:
        return Topology.REVIEW_LOOP
    if profile.complexity >= 0.25 or profile.step_count >= 3:
        return Topology.SUPERVISOR_WORKER
    return Topology.SINGLE_AGENT


def explain_topology(profile: TaskProfile, topology: Topology) -> str:
    if topology == Topology.SINGLE_AGENT:
        return "Low complexity: use direct execution plus a lightweight verification pass."
    if topology == Topology.SUPERVISOR_WORKER:
        return "Medium complexity: use a planner to define the work order before execution and verification."
    return "High complexity or risk: use an execution-review-revision loop before final synthesis."
