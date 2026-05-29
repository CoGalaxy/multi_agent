from verifiable_multi_agent.backends import LlmBackend
from verifiable_multi_agent.orchestrator import Orchestrator


class RecordingBackend(LlmBackend):
    def __init__(self) -> None:
        self.calls = []

    def complete(self, system: str, user: str, role: str | None = None) -> str:
        self.calls.append({"system": system, "user": user, "role": role})
        return f"{role} grounded response"


def test_llm_calls_are_grounded_in_original_task(tmp_path) -> None:
    backend = RecordingBackend()
    task = "Plan, execute, and verify a research task."

    Orchestrator(tmp_path / "memory.jsonl", backend=backend).solve(task)

    assert backend.calls
    assert all(f"Original task: {task}" in call["user"] for call in backend.calls)
    assert all("Do not introduce a new topic" in call["system"] for call in backend.calls)
    assert any(call["role"] == "synthesizer" for call in backend.calls)
