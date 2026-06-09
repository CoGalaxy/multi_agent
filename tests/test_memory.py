from pathlib import Path

from verifiable_multi_agent.memory import JsonlProtocolMemory
from verifiable_multi_agent.contracts import AgentRole
from verifiable_multi_agent.orchestrator import Orchestrator


def test_memory_retrieve_handles_tied_scores(tmp_path: Path) -> None:
    memory = JsonlProtocolMemory(tmp_path / "memory.jsonl")
    memory.path.write_text(
        "\n".join(
            [
                '{"task": "compare agent baselines", "topology": "single_agent", "support_rate": 1.0, "roles": [], "actions": []}',
                '{"task": "compare agent methods", "topology": "review_loop", "support_rate": 1.0, "roles": [], "actions": []}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    results = memory.retrieve("compare agent", limit=2)

    assert len(results) == 2


def test_protocol_memory_injects_memory_role_not_planner(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    memory_path.write_text(
        '{"task": "比较 AutoGen 和 CAMEL", "topology": "supervisor_worker", "support_rate": 1.0, "roles": [], "actions": []}\n',
        encoding="utf-8",
    )

    trace = Orchestrator(memory_path).solve("比较 AutoGen 和 CAMEL 的优缺点。")

    assert trace.messages[0].role == AgentRole.MEMORY
    assert trace.messages[0].subtask == "检索相似历史协作协议"


# ── ProtocolMemory 测试 ───────────────────────────────────────────

from verifiable_multi_agent.memory import MemoryRecord, ProtocolMemory


def _sample_record(task_id: str, complexity: float, verifiability: float,
                   topology: str = "single_agent", accepted: bool = True,
                   support_rate: float = 1.0) -> MemoryRecord:
    return MemoryRecord(
        task_id=task_id,
        complexity=complexity,
        verifiability=verifiability,
        topology_used=topology,
        accepted=accepted,
        support_rate=support_rate,
    )


def test_protocol_memory_add_and_query(tmp_path) -> None:
    """add() 追加记录后 query() 能检索到最近邻。"""
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    memory.add(_sample_record("t1", 0.1, 0.1))
    memory.add(_sample_record("t2", 0.8, 0.8))
    memory.add(_sample_record("t3", 0.15, 0.12))

    neighbors = memory.query(complexity=0.12, verifiability=0.11, k=2)

    assert len(neighbors) == 2
    # t1 (0.1, 0.1) 应比 t2 (0.8, 0.8) 更近
    assert neighbors[0].task_id == "t1"
    assert neighbors[1].task_id == "t3"


def test_protocol_memory_persist_and_reload(tmp_path) -> None:
    """persist: 记录持久化到 JSON 文件；reload: 新实例从文件加载。"""
    path = str(tmp_path / "memory.json")
    memory_a = ProtocolMemory(path)
    memory_a.add(_sample_record("t1", 0.3, 0.6, "review_loop", False, 0.5))

    # 新实例从同一文件加载
    memory_b = ProtocolMemory(path)
    neighbors = memory_b.query(0.3, 0.6, k=1)

    assert len(neighbors) == 1
    assert neighbors[0].task_id == "t1"
    assert neighbors[0].topology_used == "review_loop"
    assert neighbors[0].accepted is False
    assert neighbors[0].support_rate == 0.5


def test_protocol_memory_query_empty_returns_empty(tmp_path) -> None:
    """空 records 时 query 返回空列表。"""
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    assert memory.query(0.5, 0.5) == []


def test_protocol_memory_historical_fail_rate(tmp_path) -> None:
    """historical_fail_rate 正确计算邻居中 accepted=False 的比例。"""
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    memory.add(_sample_record("t1", 0.1, 0.1, accepted=True))
    memory.add(_sample_record("t2", 0.12, 0.11, accepted=False))
    memory.add(_sample_record("t3", 0.09, 0.13, accepted=False))

    neighbors = memory.query(0.1, 0.1, k=3)
    assert memory.historical_fail_rate(neighbors) == 0.667

    # 空列表返回 0.0
    assert memory.historical_fail_rate([]) == 0.0


# ── Memory 修正集成测试 ──────────────────────────────────────────

from verifiable_multi_agent.contracts import TaskProfile, Topology
from verifiable_multi_agent.router import select_topology, select_topology_with_memory


def test_select_topology_without_memory_same_as_baseline() -> None:
    """memory=None 时 select_topology_with_memory 与 select_topology 表现一致。"""
    profiles = [
        TaskProfile(task="simple", complexity=0.1, verifiability=0.1),
        TaskProfile(task="medium", complexity=0.5, verifiability=0.2),
        TaskProfile(task="hard", complexity=0.5, verifiability=0.6),
    ]
    for p in profiles:
        topo_base = select_topology(p)
        topo_mem, reasons = select_topology_with_memory(p, memory=None)
        assert topo_mem == topo_base
        assert len(reasons) == 1  # 仅基础 reason，无修正


def test_memory_correction_upgrades_single_agent(tmp_path) -> None:
    """历史邻居失败率 > 0.5 → SINGLE_AGENT 升级为 SUPERVISOR_WORKER。"""
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    # 在 (0.15, 0.15) 附近塞入两条失败记录
    memory.add(_sample_record("t1", 0.12, 0.14, "single_agent", accepted=False))
    memory.add(_sample_record("t2", 0.16, 0.13, "single_agent", accepted=False))
    memory.add(_sample_record("t3", 0.14, 0.16, "single_agent", accepted=True))

    profile = TaskProfile(task="simple task", complexity=0.15, verifiability=0.15)
    topo, reasons = select_topology_with_memory(profile, memory=memory)

    assert topo == Topology.SUPERVISOR_WORKER
    assert any("memory correction" in r for r in reasons)
    assert any("fail rate=0.667" in r for r in reasons)


def test_memory_correction_upgrades_supervisor_worker(tmp_path) -> None:
    """历史邻居失败率 > 0.5 → SUPERVISOR_WORKER 升级为 REVIEW_LOOP。"""
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    # 在 (0.5, 0.2) 附近塞入失败记录
    memory.add(_sample_record("t1", 0.48, 0.22, "supervisor_worker", accepted=False))
    memory.add(_sample_record("t2", 0.52, 0.18, "supervisor_worker", accepted=False))

    profile = TaskProfile(task="complex task", complexity=0.5, verifiability=0.2)
    topo, reasons = select_topology_with_memory(profile, memory=memory)

    assert topo == Topology.REVIEW_LOOP
    assert any("upgraded to REVIEW_LOOP" in r for r in reasons)


def test_memory_correction_no_upgrade_when_already_review_loop(tmp_path) -> None:
    """已经是 REVIEW_LOOP 时，即使失败率高也不升级（已是最高级）。"""
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    memory.add(_sample_record("t1", 0.7, 0.7, "review_loop", accepted=False))
    memory.add(_sample_record("t2", 0.72, 0.68, "review_loop", accepted=False))

    profile = TaskProfile(task="very hard task", complexity=0.7, verifiability=0.7)
    topo, reasons = select_topology_with_memory(profile, memory=memory)

    assert topo == Topology.REVIEW_LOOP
    assert not any("memory correction" in r for r in reasons)


def test_memory_correction_no_upgrade_when_low_fail_rate(tmp_path) -> None:
    """邻居失败率 <= 0.5 时不触发升级。"""
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    memory.add(_sample_record("t1", 0.12, 0.14, "single_agent", accepted=True))
    memory.add(_sample_record("t2", 0.16, 0.13, "single_agent", accepted=True))
    memory.add(_sample_record("t3", 0.14, 0.16, "single_agent", accepted=False))

    profile = TaskProfile(task="simple task", complexity=0.15, verifiability=0.15)
    topo, reasons = select_topology_with_memory(profile, memory=memory)

    # fail rate = 1/3 = 0.333 <= 0.5 → 不升级
    assert topo == Topology.SINGLE_AGENT
    assert not any("memory correction" in r for r in reasons)
