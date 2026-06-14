from pathlib import Path

from scripts.run_batch_eval import _error_row, _summary_row


def test_batch_summary_parses_multiline_generated_topology() -> None:
    output = """[Generated Topology]
Planner -> Coder -> Tester -> Reviser -> SafetyVerifier ->
Verifier -> Synthesizer
blocked=False
[Graph Execution]
review_loops_used=1
[Topology]
selected=REVIEW_LOOP
[Quantitative Router]
max_nodes=7 | max_edges=7 | max_review_loops=1 | max_tool_calls=0
[Contract Report]
support_rate=0.57
task_coverage=0.80
accepted=False
violations=['verifier_rejected:test']
"""

    row = _summary_row(
        {"id": "case", "category": "算法代码"},
        output,
        Path("runs/batch_eval_outputs/01_case.txt"),
    )

    assert row["status"] == "model_quality_failed"
    assert row["nodes"] == "Planner -> Coder -> Tester -> Reviser -> SafetyVerifier -> Verifier -> Synthesizer"
    assert row["topology"] == "review_loop"
    assert row["review_loops_used"] == 1
    assert row["max_review_loops"] == 1


def test_batch_error_row_classifies_backend_error() -> None:
    stderr = "RuntimeError: LLM backend request failed after 3 attempts: ConnectError SSL"

    row = _error_row(
        {"id": "case", "category": "接口设计"},
        1,
        stderr,
        Path("runs/batch_eval_outputs/01_case.txt"),
    )

    assert row["status"] == "backend_error"
    assert row["accepted"] is False
