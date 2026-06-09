"""
自适应拓扑路由 — 根据任务画像 (complexity, verifiability) 选择协作拓扑。

决策矩阵（二维阈值均为 0.4）：
  complexity < 0.4  且 verifiability < 0.4  → SINGLE_AGENT
  complexity >= 0.4 且 verifiability < 0.4  → SUPERVISOR_WORKER
  其他（verifiability >= 0.4）              → REVIEW_LOOP

支持 ProtocolMemory 修正：检索历史相似任务，若邻居失败率 > 0.5
且当前拓扑不是 REVIEW_LOOP，则自动升级一级拓扑。

阈值目前是经验值，后续在 benchmark 上做网格搜索调优
可以作为实验的一部分（超参数敏感性分析）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from verifiable_multi_agent.contracts import TaskProfile, Topology

if TYPE_CHECKING:
    from verifiable_multi_agent.memory import ProtocolMemory

# 拓扑升级映射
_TOPOLOGY_UPGRADE = {
    Topology.SINGLE_AGENT: Topology.SUPERVISOR_WORKER,
    Topology.SUPERVISOR_WORKER: Topology.REVIEW_LOOP,
}


def select_topology(profile: TaskProfile) -> Topology:
    """基础拓扑选择 — 仅基于任务画像的二维决策矩阵。"""
    if profile.complexity < 0.4 and profile.verifiability < 0.4:
        return Topology.SINGLE_AGENT
    if profile.complexity >= 0.4 and profile.verifiability < 0.4:
        return Topology.SUPERVISOR_WORKER
    return Topology.REVIEW_LOOP


def select_topology_with_memory(
    profile: TaskProfile,
    memory: ProtocolMemory | None = None,
) -> tuple[Topology, list[str]]:
    """带记忆修正的拓扑选择。

    1. 先用决策矩阵选出基础拓扑。
    2. 若 memory 不为 None，检索历史相似任务：
       - 若邻居失败率 > 0.5 且当前拓扑不是 REVIEW_LOOP → 升级一级
       - 在 reasons 中记录修正原因。

    Args:
        profile: 任务画像（含 complexity 和 verifiability）。
        memory: 可选的路由决策记忆，为 None 则不启用修正。

    Returns:
        (选中的拓扑, generation_reasons 列表)。
    """
    topology = select_topology(profile)
    reasons = [
        f"complexity={profile.complexity:.2f}, "
        f"verifiability={profile.verifiability:.2f} → "
        f"{topology.value.upper()}",
    ]

    if memory is None:
        return topology, reasons

    neighbors = memory.query(profile.complexity, profile.verifiability, k=3)
    if not neighbors:
        return topology, reasons

    fail_rate = memory.historical_fail_rate(neighbors)
    if fail_rate > 0.5 and topology != Topology.REVIEW_LOOP:
        upgraded = _TOPOLOGY_UPGRADE[topology]
        reasons.append(
            f"memory correction: historical fail rate={fail_rate:.3f}, "
            f"upgraded to {upgraded.value.upper()}"
        )
        return upgraded, reasons

    return topology, reasons


def explain_topology(
    profile: TaskProfile, topology: Topology, reasons: list[str] | None = None
) -> str:
    """生成拓扑选择的人类可读解释。

    Args:
        profile: 任务画像。
        topology: 选中的拓扑。
        reasons: 可选的 generation_reasons 列表（来自 select_topology_with_memory）。

    Returns:
        格式化的解释字符串，如 "complexity=0.72, verifiability=0.65 → REVIEW_LOOP"。
    """
    if reasons:
        reason = "; ".join(reasons)
    else:
        reason = (
            f"complexity={profile.complexity:.2f}, "
            f"verifiability={profile.verifiability:.2f} → "
            f"{topology.value.upper()}"
        )
    if _contains_cjk(profile.task):
        if topology == Topology.SINGLE_AGENT:
            zh = "低复杂度且低验证难度：采用直接执行，并追加轻量验证。"
        elif topology == Topology.SUPERVISOR_WORKER:
            zh = "中等复杂度：先由规划角色定义工作顺序，再执行并验证。"
        else:
            zh = "高验证难度：采用执行-审查-修订闭环，再进行最终合成。"
        return f"{reason}（{zh}）"
    return reason


def _contains_cjk(text: str) -> bool:
    return any("一" <= char <= "鿿" for char in text)
