from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifiable_multi_agent.backends import DeepSeekBackend
from verifiable_multi_agent.cli import load_env_file
from verifiable_multi_agent.orchestrator import Orchestrator
from verifiable_multi_agent.routing_memory import ProtocolMemory
from verifiable_multi_agent.trace import build_run_trace

from judge_prompts import BASELINE_SYSTEM_PROMPT


def main() -> int:
    parser = argparse.ArgumentParser(description="Run contract message effectiveness experiment.")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-contract", action="store_true")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is not set. Put it in .env or the environment.", file=sys.stderr)
        return 2

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = build_deepseek_backend(api_key, args.model, args.judge_model)
    raw_path = args.output_dir / "raw_runs.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['id']} {case['category']}")
            if not args.skip_baseline:
                write_jsonl(handle, run_baseline(case, backend))
            if not args.skip_contract:
                write_jsonl(handle, run_contract(case, backend, args.output_dir))

    print(f"Wrote raw runs: {raw_path}")
    return 0


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def build_deepseek_backend(api_key: str, model: str | None, judge_model: str | None) -> DeepSeekBackend:
    return DeepSeekBackend(
        api_key=api_key,
        model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        judge_model=judge_model or os.getenv("DEEPSEEK_JUDGE_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def run_baseline(case: dict[str, Any], backend: DeepSeekBackend) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        answer = backend.complete(system=BASELINE_SYSTEM_PROMPT, user=case["task"], role="baseline")
        return {
            "case_id": case["id"],
            "category": case["category"],
            "group": "baseline",
            "status": "ok",
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "task": case["task"],
            "answer": answer,
            "required_deliverables": case["required_deliverables"],
            "reference_facts": case["reference_facts"],
            "hallucination_risks": case["hallucination_risks"],
        }
    except Exception as exc:  # noqa: BLE001 - experiment must keep running.
        return error_row(case, "baseline", exc, started)


def run_contract(case: dict[str, Any], backend: DeepSeekBackend, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        trace = Orchestrator(
            memory_path=output_dir / "protocol_memory.jsonl",
            backend=backend,
            routing_memory=ProtocolMemory(str(output_dir / "routing_memory.json")),
        ).solve(case["task"])
        run_trace = build_run_trace(trace)
        verification = run_trace.verification_result or {}
        report = run_trace.contract_report.model_dump(mode="json")
        return {
            "case_id": case["id"],
            "category": case["category"],
            "group": "contract",
            "status": "ok",
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "task": case["task"],
            "answer": run_trace.final_answer or "",
            "accepted": verification.get("accepted"),
            "task_coverage": verification.get("task_coverage"),
            "support_rate": verification.get("support_rate"),
            "violations": verification.get("violations", []),
            "contract_report": report,
            "trace": run_trace.model_dump(mode="json"),
            "required_deliverables": case["required_deliverables"],
            "reference_facts": case["reference_facts"],
            "hallucination_risks": case["hallucination_risks"],
        }
    except Exception as exc:  # noqa: BLE001 - experiment must keep running.
        return error_row(case, "contract", exc, started)


def error_row(case: dict[str, Any], group: str, exc: Exception, started: float) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "category": case["category"],
        "group": group,
        "status": error_status(str(exc)),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "task": case["task"],
        "answer": "",
        "error": str(exc),
        "required_deliverables": case.get("required_deliverables", []),
        "reference_facts": case.get("reference_facts", []),
        "hallucination_risks": case.get("hallucination_risks", []),
    }


def error_status(message: str) -> str:
    text = message.lower()
    if any(marker in text for marker in ("backend", "connect", "timeout", "httpx", "ssl", "deepseek")):
        return "backend_error"
    return "runtime_error"


def write_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


if __name__ == "__main__":
    raise SystemExit(main())
