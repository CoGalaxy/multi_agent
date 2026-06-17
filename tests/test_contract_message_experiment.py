from __future__ import annotations

import importlib.util
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "contract_message_effectiveness"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_message_cases_are_valid_jsonl() -> None:
    run_experiment = _load_module("contract_run_experiment", "run_experiment.py")

    cases = run_experiment.load_cases(EXPERIMENT_DIR / "cases.jsonl")

    assert len(cases) == 24
    assert {case["category"] for case in cases} == {
        "code_test",
        "math_proof",
        "db_api",
        "fact_grounded",
    }
    assert all(case["required_deliverables"] for case in cases)
    assert all(case["reference_facts"] for case in cases)
    assert all(case["hallucination_risks"] for case in cases)


def test_contract_message_summary_counts_hallucination_detection() -> None:
    evaluator = _load_module("contract_evaluate_results", "evaluate_results.py")
    rows = [
        {
            "case_id": "a",
            "category": "code_test",
            "group": "baseline",
            "status": "ok",
            "answer": "已完成代码和测试。",
            "required_deliverables": ["Python 实现代码", "测试用例"],
            "reference_facts": [],
            "hallucination_risks": [],
        },
        {
            "case_id": "a",
            "category": "code_test",
            "group": "contract",
            "status": "ok",
            "answer": "已完成代码和测试。",
            "accepted": False,
            "task_coverage": 0.5,
            "violations": ["missing_task_coverage:code"],
            "required_deliverables": ["Python 实现代码", "测试用例"],
            "reference_facts": [],
            "hallucination_risks": [],
        },
    ]

    evaluated = evaluator.evaluate_records(rows, judge_backend=None)
    summary = evaluator.summarize(evaluated)

    assert summary["groups"]["baseline"]["count"] == 1
    assert summary["groups"]["contract"]["count"] == 1
    assert summary["groups"]["contract"]["hallucination_detection_rate"] == 1.0
    assert summary["groups"]["contract"]["average_task_coverage"] == 0.5


def test_contract_message_error_status_separates_backend_errors() -> None:
    run_experiment = _load_module("contract_run_experiment_error", "run_experiment.py")

    assert run_experiment.error_status("DeepSeek backend request failed after timeout") == "backend_error"
    assert run_experiment.error_status("KeyError: missing field") == "runtime_error"
