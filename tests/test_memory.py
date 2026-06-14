from pathlib import Path

from verifiable_multi_agent.contracts import TaskProfile, Topology
from verifiable_multi_agent.memory import JsonlProtocolMemory, MemoryRecord, ProtocolMemory
from verifiable_multi_agent.quantitative_router import (
    QuantitativeRouter,
    infer_input_requirements,
    topology_from_spec,
)


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


def _sample_record(
    task_id: str,
    complexity: float,
    verifiability: float,
    topology: str = "single_agent",
    accepted: bool = True,
    support_rate: float = 1.0,
) -> MemoryRecord:
    return MemoryRecord(
        task_id=task_id,
        complexity=complexity,
        verifiability=verifiability,
        topology_used=topology,
        accepted=accepted,
        support_rate=support_rate,
    )


def test_protocol_memory_add_and_query(tmp_path) -> None:
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    memory.add(_sample_record("t1", 0.1, 0.1))
    memory.add(_sample_record("t2", 0.8, 0.8))
    memory.add(_sample_record("t3", 0.15, 0.12))

    neighbors = memory.query(complexity=0.12, verifiability=0.11, k=2)

    assert len(neighbors) == 2
    assert neighbors[0].task_id == "t1"
    assert neighbors[1].task_id == "t3"


def test_protocol_memory_persist_and_reload(tmp_path) -> None:
    path = str(tmp_path / "memory.json")
    memory_a = ProtocolMemory(path)
    memory_a.add(_sample_record("t1", 0.3, 0.6, "review_loop", False, 0.5))

    memory_b = ProtocolMemory(path)
    neighbors = memory_b.query(0.3, 0.6, k=1)

    assert len(neighbors) == 1
    assert neighbors[0].task_id == "t1"
    assert neighbors[0].topology_used == "review_loop"
    assert neighbors[0].accepted is False
    assert neighbors[0].support_rate == 0.5


def test_protocol_memory_query_empty_returns_empty(tmp_path) -> None:
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    assert memory.query(0.5, 0.5) == []


def test_protocol_memory_historical_fail_rate(tmp_path) -> None:
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    memory.add(_sample_record("t1", 0.1, 0.1, accepted=True))
    memory.add(_sample_record("t2", 0.12, 0.11, accepted=False))
    memory.add(_sample_record("t3", 0.09, 0.13, accepted=False))

    neighbors = memory.query(0.1, 0.1, k=3)
    assert memory.historical_fail_rate(neighbors) == 0.667
    assert memory.historical_fail_rate([]) == 0.0


def _route_with_memory(profile: TaskProfile, memory: ProtocolMemory | None = None) -> Topology:
    spec = QuantitativeRouter().route(
        profile,
        infer_input_requirements(profile.task),
        memory=memory,
    )
    return topology_from_spec(spec)


def test_quant_router_without_memory_uses_matrix() -> None:
    profiles = [
        (TaskProfile(task="simple", complexity=0.1, verifiability=0.1), Topology.SINGLE_AGENT),
        (TaskProfile(task="medium", complexity=0.5, verifiability=0.2), Topology.SUPERVISOR_WORKER),
        (TaskProfile(task="hard", complexity=0.5, verifiability=0.6), Topology.REVIEW_LOOP),
    ]
    for profile, expected in profiles:
        assert _route_with_memory(profile, memory=None) == expected


def test_memory_correction_upgrades_capabilities_for_single_agent(tmp_path) -> None:
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    memory.add(_sample_record("t1", 0.12, 0.14, "single_agent", accepted=False))
    memory.add(_sample_record("t2", 0.16, 0.13, "single_agent", accepted=False))
    memory.add(_sample_record("t3", 0.14, 0.16, "single_agent", accepted=True))

    profile = TaskProfile(task="simple task", complexity=0.15, verifiability=0.15)
    spec = QuantitativeRouter().route(profile, infer_input_requirements(profile.task), memory=memory)

    assert spec.capability_needs.planning
    assert spec.capability_needs.synthesis
    assert any("memory correction" in reason for reason in spec.generation_reasons)


def test_memory_correction_no_upgrade_when_already_review_loop(tmp_path) -> None:
    memory = ProtocolMemory(str(tmp_path / "memory.json"))
    memory.add(_sample_record("t1", 0.7, 0.7, "review_loop", accepted=False))
    memory.add(_sample_record("t2", 0.72, 0.68, "review_loop", accepted=False))

    profile = TaskProfile(task="very hard task", complexity=0.7, verifiability=0.7)
    spec = QuantitativeRouter().route(profile, infer_input_requirements(profile.task), memory=memory)

    assert topology_from_spec(spec) == Topology.REVIEW_LOOP
    assert not any("memory correction" in reason for reason in spec.generation_reasons)
