from pathlib import Path
import json

from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.contracts import Topology
from verifiable_multi_agent.orchestrator import Orchestrator

from tests.fake_llm import FakeLlmBackend


class RejectThenApproveBackend(LlmBackend):
    is_real_llm = True

    def __init__(self) -> None:
        self.verifier_calls = 0
        self.reviser_prompts: list[str] = []

    def complete(self, system: str, user: str, role: str | None = None) -> str:
        if "Subtask: Revise candidate solution" in user:
            self.reviser_prompts.append(user)
        if role == "verifier":
            self.verifier_calls += 1
            if self.verifier_calls <= 2:
                return json.dumps(
                    {
                        "claim": "REJECTED: executor claim conflicts with evidence item",
                        "evidence": ["executor claim conflicts with evidence item"],
                        "action": "verify",
                        "uncertainty": 0.1,
                    }
                )
            return json.dumps(
                {
                    "claim": "APPROVED verification passed",
                    "evidence": ["APPROVED verification passed"],
                    "action": "verify",
                    "uncertainty": 0.1,
                }
            )
        claim = f"{role or 'agent'} completed supported answer"
        return json.dumps(
            {
                "claim": claim,
                "evidence": [claim],
                "action": role or "agent",
                "uncertainty": 0.1,
            }
        )


def test_simple_task_uses_small_topology(tmp_path: Path) -> None:
    trace = Orchestrator(tmp_path / "memory.jsonl", backend=FakeLlmBackend()).solve("Summarize this task.")

    assert trace.topology == Topology.SINGLE_AGENT
    assert trace.metadata["router_mode"] == "quant"
    assert trace.verification is not None
    assert trace.final_answer


def test_risky_task_uses_review_loop(tmp_path: Path) -> None:
    task = "Investigate a security task, verify evidence, then write a safe answer."
    trace = Orchestrator(tmp_path / "memory.jsonl", backend=FakeLlmBackend()).solve(task)

    assert trace.topology == Topology.REVIEW_LOOP
    assert trace.profile.verifiability >= 0.4
    assert trace.verification is not None
    assert trace.final_answer


def test_protocol_memory_is_written(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    Orchestrator(memory_path, backend=FakeLlmBackend()).solve("Respond briefly.")

    assert memory_path.exists()
    assert memory_path.read_text(encoding="utf-8").strip()


def test_verifier_rejection_upgrades_to_review_loop(tmp_path: Path) -> None:
    backend = RejectThenApproveBackend()
    trace = Orchestrator(tmp_path / "memory.jsonl", backend=backend).solve("Summarize this task.")

    assert backend.verifier_calls >= 3
    assert trace.topology == Topology.REVIEW_LOOP
    assert trace.metadata["review_loop_upgrade_reason"] == "verifier_rejected"
    assert "reviser" in trace.metadata["graph_execution"]["executed_nodes"]
    assert backend.reviser_prompts
    assert "verifier_rejected" in backend.reviser_prompts[-1]
    assert "REJECTED: executor claim conflicts with evidence item" in backend.reviser_prompts[-1]
